#!/usr/bin/env python3
"""
ollama-layer-tuner: find the num_gpu (layers offloaded to the GPU) that gives the fastest generation
for an Ollama model on *your* machine.

Ollama picks the number of GPU layers automatically. On small GPUs (4-8 GB) that guess is often far from the
fastest setting, in either direction. This script measures it instead of guessing:

  1. reads the layer count of the model,
  2. measures generation speed with Ollama's automatic choice (num_gpu = -1),
  3. coarse sweep from 0 to all layers, then a finer search around the best point,
  4. re-measures the two best settings and averages them (single runs are noisy by ~5%),
  5. prints the result and, with --apply, creates a tuned copy of the model.

Only the Python standard library is used. Every measurement loads the model from scratch, so other models are
unloaded first and nothing else should be using the GPU.

  python tune_layers.py                          # pick a model from a list
  python tune_layers.py qwen2.5:7b-instruct --ctx 8192
  python tune_layers.py qwen2.5:7b-instruct --apply --name qwen2.5-7b-fast
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Explain in simple words why the sky is blue. Write 5-6 sentences."


class Ollama:
    def __init__(self, host):
        self.host = host.rstrip("/")

    def call(self, path, body=None, timeout=1800):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.host + path, data, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def cli(self, *args):
        env = {**os.environ, "OLLAMA_HOST": self.host.replace("http://", "")}
        return subprocess.run(["ollama", *args], env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")

    def unload_all(self):
        for m in self.call("/api/ps").get("models", []):
            self.call("/api/generate", {"model": m["name"], "keep_alive": 0})


def layer_count(ol, model):
    info = ol.call("/api/show", {"model": model}).get("model_info", {})
    n = next((v for k, v in info.items() if k.endswith(".block_count")), None)
    return int(n) if n else None


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def measure(ol, model, ng, ctx, tokens, log):
    """Generation speed (tokens/s) with num_gpu = ng (-1 = Ollama decides). None if the model did not load."""
    opts = {"temperature": 0, "seed": 1, "num_predict": tokens, "num_ctx": ctx, "num_gpu": ng}
    body = {"model": model, "prompt": PROMPT, "stream": False, "think": False, "options": opts, "keep_alive": "5m"}
    label = "auto" if ng == -1 else str(ng)
    res = None
    try:
        ol.call("/api/generate", {**body, "prompt": "Hi", "options": {**opts, "num_predict": 1}})   # load, not timed
        ps = ol.call("/api/ps").get("models", [])
        vram = next((m.get("size_vram", 0) for m in ps if m["name"] in (model, model + ":latest")), 0)
        r = ol.call("/api/generate", body)
        res = {"tok_s": r["eval_count"] / (r["eval_duration"] / 1e9),
               "prompt_tok_s": r.get("prompt_eval_count", 0) / max(r.get("prompt_eval_duration", 1) / 1e9, 1e-9),
               "vram_gb": vram / 2**30}
        log(f"  num_gpu {label:>4}  {res['tok_s']:7.2f} tok/s   GPU memory {res['vram_gb']:.2f} GB")
    except (urllib.error.URLError, OSError, KeyError, ZeroDivisionError, ValueError) as e:
        log(f"  num_gpu {label:>4}  failed (does not fit?): {str(e)[:90]}")
    try:
        ol.call("/api/generate", {"model": model, "keep_alive": 0})   # next measurement starts from a clean GPU
    except OSError:
        pass
    time.sleep(2)
    return res


def tune(ol, model, ctx, tokens, log):
    total = layer_count(ol, model)
    if not total:
        sys.exit(f"Cannot read the layer count of {model}")
    top = total + 1                                    # +1: the output layer is offloaded as well
    log(f"\n{model}: {total} layers, context {ctx}, {tokens} tokens per test")
    res = {}

    def m(ng):
        if ng not in res:
            res[ng] = measure(ol, model, ng, ctx, tokens, log)
        return res[ng]

    m(-1)
    step = max(1, round(top / 6))
    for ng in sorted({0, *range(step, top, step), top}):      # coarse sweep over the whole range
        m(ng)
    ok = lambda: [k for k, v in res.items() if k >= 0 and v]
    while step > 1 and ok():                                  # finer search around the best point
        best = max(ok(), key=lambda k: res[k]["tok_s"])
        step = max(1, step // 2)
        for ng in (best - step, best + step):
            if 0 <= ng <= top:
                m(ng)
    ranked = sorted(ok(), key=lambda k: -res[k]["tok_s"])
    if not ranked:
        sys.exit("No setting could be loaded")
    log("  re-measuring the two best settings:")
    final = {}
    for ng in ranked[:2]:
        again = measure(ol, model, ng, ctx, tokens, log)
        final[ng] = (res[ng]["tok_s"] + again["tok_s"]) / 2 if again else res[ng]["tok_s"]
    best = max(final, key=final.get)
    auto = res[-1]["tok_s"] if res.get(-1) else None
    return {"model": model, "layers": total, "ctx": ctx, "best_num_gpu": best, "best_tok_s": round(final[best], 2),
            "auto_tok_s": auto and round(auto, 2), "speedup": auto and round(final[best] / auto, 2),
            "sweep": {("auto" if k == -1 else str(k)): v and {kk: round(vv, 2) for kk, vv in v.items()}
                      for k, v in sorted(res.items())}}


def apply(ol, model, ng, ctx, name, log):
    mf = ol.cli("show", "--modelfile", model)
    if mf.returncode:
        log("Could not read the Modelfile: " + mf.stderr.strip())
        return
    text = re.sub(r"(?im)^PARAMETER\s+(num_gpu|num_ctx)\s+.*\n?", "", mf.stdout).rstrip()
    text += f"\nPARAMETER num_gpu {ng}\nPARAMETER num_ctx {ctx}\n"
    path = Path(f"Modelfile.{re.sub(r'[^A-Za-z0-9._-]+', '_', name)}")
    path.write_text(text, encoding="utf-8")
    r = ol.cli("create", name, "-f", str(path))
    log(f"Created model '{name}' (Modelfile: {path})" if r.returncode == 0 else "ollama create failed: " + r.stderr[-300:])


def default_name(model, ctx):
    """qwen3:4b-instruct -> qwen3-4b-instruct-tuned; other context lengths get their own copy (-32k-tuned)."""
    base = re.sub(r":latest$", "", model.split("/")[-1])
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-")
    return base + ("" if ctx == 8192 else f"-{ctx // 1024}k") + "-tuned"


def main():
    ap = argparse.ArgumentParser(description="Find the fastest num_gpu for an Ollama model on this machine.")
    ap.add_argument("model", nargs="?", help="model name; omit to choose from a list")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                    help="Ollama server (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
    ap.add_argument("--ctx", type=int, default=8192, help="context length you will actually use (affects GPU memory)")
    ap.add_argument("--tokens", type=int, default=64, help="tokens generated per measurement")
    ap.add_argument("--apply", action="store_true", help="create a tuned copy of the model with the best num_gpu")
    ap.add_argument("--name", help="name for the tuned copy (default: <model>-<tag>-tuned)")
    ap.add_argument("--out", default="results.json", help="JSON file to append results to")
    a = ap.parse_args()
    if not a.host.startswith("http"):
        a.host = "http://" + a.host

    ol = Ollama(a.host)
    try:
        names = sorted(m["name"] for m in ol.call("/api/tags")["models"])
    except OSError as e:
        sys.exit(f"Ollama is not reachable at {a.host}: {e}")
    model = a.model
    if not model:
        if not names:
            sys.exit(f"No models on {a.host}. Pull one first, e.g.: ollama pull llama3.2:3b")
        for i, n in enumerate(names, 1):
            print(f"  {i:2}. {n}")
        while not model:
            s = input("Model number: ").strip()
            if s.isdigit() and 1 <= int(s) <= len(names):
                model = names[int(s) - 1]

    def log(s):
        print(s, flush=True)

    ol.unload_all()
    r = tune(ol, model, a.ctx, a.tokens, log)
    r.update({"date": time.strftime("%Y-%m-%d"), "gpu": gpu_name()})
    log(f"\nBEST: num_gpu {r['best_num_gpu']} -> {r['best_tok_s']} tok/s"
        + (f"   (Ollama auto: {r['auto_tok_s']} tok/s, x{r['speedup']})" if r["auto_tok_s"] else ""))
    log(f"To use it: add 'PARAMETER num_gpu {r['best_num_gpu']}' to the Modelfile, "
        f"or pass options {{\"num_gpu\": {r['best_num_gpu']}}} in API calls.")

    out = Path(a.out)
    try:
        allres = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    except ValueError:
        backup = out.with_suffix(".broken.json")
        out.replace(backup)
        log(f"Warning: {out} was not valid JSON; it was kept as {backup} and a new file was started.")
        allres = []
    allres = [x for x in allres if not (x.get("model") == model and x.get("ctx") == a.ctx)] + [r]
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(allres, indent=1), encoding="utf-8")
    tmp.replace(out)

    if a.apply:
        apply(ol, model, r["best_num_gpu"], a.ctx, a.name or default_name(model, a.ctx), log)


if __name__ == "__main__":
    main()
