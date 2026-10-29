#!/usr/bin/env python3
"""Print a Markdown table from results.json (or another file): python report.py [results.json]"""

import json
import sys

LABELS = {  # local names of fine-tunes used in the benchmark -> what they are
    "unc9b": "Qwen3.5-9B Q4_K_M (HauhauCS fine-tune)",
    "unc14b": "Qwen3-14B Q4_K_M (Huihui fine-tune)",
}


def main():
    rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json", encoding="utf-8"))
    print("| Model | Layers | Context | Ollama auto, tok/s | Best num_gpu | Best, tok/s | Speed-up |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        auto = f"{r['auto_tok_s']:.1f}" if r.get("auto_tok_s") else "failed"
        sp = f"x{r['speedup']:.2f}" if r.get("speedup") else "-"
        name = LABELS.get(r["model"].replace(":latest", ""), r["model"])
        print(f"| {name} | {r['layers']} | {r['ctx']} | {auto} | {r['best_num_gpu']} | {r['best_tok_s']:.1f} | {sp} |")


if __name__ == "__main__":
    main()
