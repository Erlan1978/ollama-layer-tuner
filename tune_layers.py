#!/usr/bin/env python3
"""
ollama-layer-tuner: find the num_gpu (layers offloaded to the GPU) that gives the fastest generation
for an Ollama model on *your* machine.

Ollama picks the number of GPU layers automatically. On small GPUs (4-8 GB) that guess is often far from the
fastest setting, in either direction. This script measures it instead of guessing:

  1. reads the layer count of the model,
  2. measures generation speed with Ollama's automatic choice (num_gpu = -1),
  3. coarse sweep from 0 to all layers, then a finer search around the best point,
  4. runs the two best settings and Ollama auto once more and compares only those repeat runs
     (the sweep winner's first number is too high: it is the best of many noisy runs),
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

DEFAULT_CTX = 8192
PROMPT = "Explain in simple words why the sky is blue. Write 5-6 sentences."


def normalize_host(h):
    """Read OLLAMA_HOST the way the ollama CLI does: '0.0.0.0', ':11434', 'pc', 'pc:11434', 'https://pc'.

    0.0.0.0 and :: mean "listen on every interface" (the Ollama FAQ recommends it for LAN access) and cannot be
    connected to, so the local address is used. localhost becomes 127.0.0.1: on Windows 'localhost' tries IPv6
    first and adds ~2 s to every request. Without a port: 11434, or 80/443 when the scheme is given (as Ollama does).
    """
    h = (h or "").strip()
    scheme, port = "http", "11434"
    if "://" in h:
        scheme, h = h.split("://", 1)
        port = {"http": "80", "https": "443"}.get(scheme, port)
    h = h.split("/", 1)[0]
    if h.startswith("["):                                   # [::1]:11434
        host, _, rest = h[1:].partition("]")
        port = rest.lstrip(":") or port
    elif h.count(":") > 1:                                  # bare IPv6 address
        host = h
    else:
        host, _, p = h.partition(":")
        port = p or port
    if host in ("", "0.0.0.0", "::", "localhost"):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


class Ollama:
    def __init__(self, host):
        self.host = host.rstrip("/")

    def call(self, path, body=None, timeout=1800):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.host + path, data, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def cli(self, *args):
        env = {**os.environ, "OLLAMA_HOST": self.host}
        try:
            return subprocess.run(["ollama", *args], env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return subprocess.CompletedProcess(args, 1, "", "the 'ollama' command is not installed on this machine")

    def unload(self, model):
        try:
            self.call("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
        except OSError:
            pass

    def unload_all(self):
        for m in self.call("/api/ps").get("models", []):
            self.unload(m["name"])

    def is_local(self):
        return any(x in self.host for x in ("://127.", "://[::1]"))


def layer_count(ol, model):
    info = ol.call("/api/show", {"model": model}).get("model_info", {})
    arch = info.get("general.architecture")
    n = info.get(f"{arch}.block_count") if arch else None
    if n is None:      # vision models also have "<arch>.vision.block_count"; that is not the number we need
        n = next((v for k, v in info.items() if k.endswith(".block_count") and ".vision." not in k), None)
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
    ol.unload(model)                                   # next measurement starts from a clean GPU
    time.sleep(2)
    return res


def tune(ol, model, ctx, tokens, log, res):
    """res: dict num_gpu -> first measurement, filled as we go (the caller keeps it if the run is interrupted)."""
    total = layer_count(ol, model)
    if not total:
        sys.exit(f"Cannot read the layer count of {model}")
    top = total + 1                                    # +1: the output layer is offloaded as well
    log(f"\n{model}: {total} layers, context {ctx}, {tokens} tokens per test")

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

    # The sweep winner is the maximum of many noisy runs, so its first number is too high ("winner's curse").
    # Only repeat runs are compared, and Ollama auto is repeated too, so both sides are measured the same way.
    # A setting that does not load a second time is too close to the memory limit to recommend.
    log("  repeat runs (only these are compared):")
    repeat = {}
    for ng in ranked[:4]:
        if len(repeat) == 2:
            break
        again = measure(ol, model, ng, ctx, tokens, log)
        if again:
            repeat[ng] = again["tok_s"]
        else:
            log(f"  num_gpu {ng} did not load on the repeat run - skipped as unreliable")
    if not repeat:
        sys.exit("None of the best settings loaded a second time: the GPU is too close to full")
    best = max(repeat, key=repeat.get)
    auto_runs = [v["tok_s"] for v in (res.get(-1), measure(ol, model, -1, ctx, tokens, log)) if v]
    auto = sum(auto_runs) / len(auto_runs) if auto_runs else None
    return {"model": model, "layers": total, "ctx": ctx, "best_num_gpu": best, "best_tok_s": round(repeat[best], 2),
            "auto_tok_s": auto and round(auto, 2), "speedup": auto and round(repeat[best] / auto, 2),
            "repeat": {str(k): round(v, 2) for k, v in repeat.items()},
            "auto_runs": [round(v, 2) for v in auto_runs],
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
    if ctx != DEFAULT_CTX:
        base += f"-{ctx // 1024}k" if ctx % 1024 == 0 else f"-{ctx}"
    return base + "-tuned"


def load_results(out, log):
    """Existing results, or [] - a file that is not a JSON list is moved aside, never silently overwritten."""
    if not out.exists():
        return []
    try:
        data = json.loads(out.read_text(encoding="utf-8-sig"))
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data
    except ValueError:
        pass
    backup = out.with_name(f"{out.stem}.broken-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.replace(backup)
    log(f"Warning: {out} is not a results list; it was kept as {backup} and a new file will be started.")
    return []


def main():
    ap = argparse.ArgumentParser(description="Find the fastest num_gpu for an Ollama model on this machine.")
    ap.add_argument("model", nargs="?", help="model name; omit to choose from a list")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"),
                    help="Ollama server (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                    help="context length you will actually use (affects GPU memory)")
    ap.add_argument("--tokens", type=int, default=64, help="tokens generated per measurement")
    ap.add_argument("--apply", action="store_true", help="create a tuned copy of the model with the best num_gpu")
    ap.add_argument("--name", help="name for the tuned copy (default: <model>-<tag>-tuned)")
    ap.add_argument("--out", default="results.json",
                    help="JSON file with results (a new run of the same model and context replaces the old one)")
    a = ap.parse_args()
    if a.ctx < 256 or a.tokens < 8:
        sys.exit("--ctx must be at least 256 and --tokens at least 8")
    a.host = normalize_host(a.host)

    def log(s):
        print(s, flush=True)

    ol = Ollama(a.host)
    try:
        names = sorted(m["name"] for m in ol.call("/api/tags", timeout=15)["models"])
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"Ollama is not reachable at {a.host}: {e}")
    if not names:
        sys.exit(f"No models on {a.host}. Pull one first, e.g.: ollama pull llama3.2:3b")
    model = a.model
    if model and model not in names and model + ":latest" not in names:
        similar = [n for n in names if n.split(":")[0] == model.split(":")[0]]
        sys.exit(f"Model '{model}' is not on {a.host}."
                 + (f" Installed: {', '.join(similar)}." if similar else "")
                 + f" To download it: ollama pull {model}")
    if not model:
        for i, n in enumerate(names, 1):
            print(f"  {i:2}. {n}")
        while not model:
            try:
                s = input("Model number: ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit("\nNo model chosen")
            try:
                model = names[int(s) - 1] if 1 <= int(s) <= len(names) else None
            except ValueError:
                pass

    out = Path(a.out)                       # checked before tuning, so a long run is never lost at the end
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = load_results(out, log)

    ol.unload_all()
    res = {}
    try:
        r = tune(ol, model, a.ctx, a.tokens, log, res)
    except KeyboardInterrupt:
        ol.unload(model)
        done = {k: v["tok_s"] for k, v in res.items() if k >= 0 and v}
        if done:
            b = max(done, key=done.get)
            log(f"\nStopped. Best so far (one run, not confirmed): num_gpu {b} -> {done[b]:.2f} tok/s. Nothing saved.")
        else:
            log("\nStopped. Nothing saved.")
        sys.exit(130)
    r.update({"date": time.strftime("%Y-%m-%d"), "gpu": gpu_name() if ol.is_local() else None})
    log(f"\nBEST: num_gpu {r['best_num_gpu']} -> {r['best_tok_s']} tok/s"
        + (f"   (Ollama auto: {r['auto_tok_s']} tok/s, x{r['speedup']})" if r["auto_tok_s"] else ""))
    log(f"To use it: add 'PARAMETER num_gpu {r['best_num_gpu']}' to the Modelfile, "
        f"or pass options {{\"num_gpu\": {r['best_num_gpu']}}} in API calls.")

    allres = [x for x in allres if not (x.get("model") == model and x.get("ctx") == a.ctx)] + [r]
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(allres, indent=1), encoding="utf-8")
    tmp.replace(out)

    if a.apply:
        apply(ol, model, r["best_num_gpu"], a.ctx, a.name or default_name(model, a.ctx), log)


if __name__ == "__main__":
    main()
