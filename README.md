# ollama-layer-tuner

Find the fastest `num_gpu` (number of layers offloaded to the GPU) for any [Ollama](https://ollama.com) model
**on your own machine** — by measuring, not guessing.

On small GPUs (4–8 GB) Ollama's automatic choice can be several times slower than the best setting.
On a GTX 1650 SUPER (4 GB) the tuned setting was **1.4× to 6.8× faster** than Ollama's default for every model
tested below.

One file, Python standard library only, works with any Ollama server.

## Results

Generation speed in tokens per second (higher is better). Every number is a real measurement made with this script;
the raw data is in [`results/bench.json`](results/bench.json) and the full log in [`results/bench.log`](results/bench.log).

| Model | Layers | Context | Ollama auto, tok/s | Best `num_gpu` | Best, tok/s | Speed-up |
|---|---|---|---|---|---|---|
| llama3.2:3b | 28 | 8192 | 21.8 | 29 | 70.5 | **×3.2** |
| qwen3:4b-instruct | 36 | 8192 | 12.8 | 37 | 55.6 | **×4.4** |
| gemma3:4b | 34 | 8192 | 9.7 | 35 | 54.4 | **×5.6** |
| qwen3:4b-instruct | 36 | 32768 | 7.1 | 37 | 48.3 | **×6.8** |
| qwen2.5:7b-instruct | 28 | 8192 | 5.2 | 25 | 15.3 | **×2.9** |
| Qwen3.5-9B Q4_K_M (community fine-tune) | 32 | 8192 | 4.0 | 23 | 7.6 | **×1.9** |
| Qwen3-14B Q4_K_M (community fine-tune) | 40 | 8192 | 2.1 | 21 | 2.9 | **×1.4** |

Test machine: NVIDIA GTX 1650 SUPER 4 GB (driver 610.88), Intel i5-12400F, 16 GB DDR4 (one stick, 2400 MT/s),
Windows 11, Ollama 0.40.11. 64 generated tokens per measurement, temperature 0.

What the numbers show on this machine:
- Ollama's automatic choice put only about **2.1 GB** of every model on the 4 GB card, however small the model was.
- Small models (3–4B) run several times faster when **all** layers go to the GPU.
- For 7B–14B models the best point is in the middle: too few layers wastes the GPU, too many slows things down again.
  That point is different for every model and context length, which is why it is worth measuring.

Another machine, driver or Ollama version will give different numbers. Run the script on yours.

## Usage

Requirements: Python 3.8+, a running Ollama server, and the `ollama` command only if you use `--apply`.

```bash
python tune_layers.py                              # choose a model from a list
python tune_layers.py qwen2.5:7b-instruct          # tune one model (context 8192)
python tune_layers.py qwen3:4b-instruct --ctx 32768   # tune for the context length you really use
python tune_layers.py qwen2.5:7b-instruct --apply --name qwen2.5-7b-fast   # also create a tuned copy
python tune_layers.py llama3.2:3b --host http://192.168.1.10:11434         # another Ollama server
python report.py results.json                      # Markdown table of everything measured
```

Example output:

```
qwen2.5:7b-instruct: 28 layers, context 8192, 64 tokens per test
  num_gpu auto     5.23 tok/s   GPU memory 2.10 GB
  num_gpu    0     ...
  ...
BEST: num_gpu 25 -> 15.32 tok/s   (Ollama auto: 5.23 tok/s, x2.93)
To use it: add 'PARAMETER num_gpu 25' to the Modelfile, or pass options {"num_gpu": 25} in API calls.
```

How to use the result:
- **Modelfile:** `PARAMETER num_gpu 25`, then `ollama create my-model -f Modelfile` (or just run with `--apply`).
- **API:** `"options": {"num_gpu": 25}` in `/api/generate` or `/api/chat` requests.

### Options

| Option | Default | Meaning |
|---|---|---|
| `model` | list to choose from | model name as in `ollama list` |
| `--host` | `$OLLAMA_HOST` or `http://127.0.0.1:11434` | Ollama server |
| `--ctx` | 8192 | context length to tune for; a longer context needs more GPU memory, so the best value changes |
| `--tokens` | 64 | tokens generated per measurement |
| `--apply` | off | create a tuned copy of the model (the original is not changed) |
| `--name` | `<model>-<tag>-tuned`, e.g. `qwen3-4b-instruct-tuned` (`-32k-tuned` for `--ctx 32768`) | name of the tuned copy |
| `--out` | `results.json` | file where results are appended |

## How it works

1. Reads the number of layers from `/api/show`.
2. Measures Ollama's automatic choice (`num_gpu = -1`).
3. Coarse sweep from 0 to all layers in about 6 steps, then halves the step around the best point down to 1.
4. Re-measures the two best settings and averages them: single runs differ by up to ~5%.
5. Every measurement loads the model from scratch, generates a short answer and unloads it. Only the generation
   speed (`eval_duration`) is timed; loading and prompt reading are not.

Tuning takes about 2 minutes for a 3B model and 15–30 minutes for 14B+ models, because every step reloads the model.

## Limitations — please read

- **Close other GPU programs and other loaded models during tuning**, or the numbers will be wrong. The script unloads
  models on the server it tunes, but not on other Ollama servers.
- **"GPU memory" can exceed the card's VRAM on Windows.** In the table above, some best settings report 4–7 GB on a
  4 GB card. The NVIDIA driver on Windows can spill GPU memory into system RAM ("sysmem fallback"), and Ollama counts
  that as GPU memory. It was still the fastest setting in these tests. On Linux, or with sysmem fallback disabled,
  such settings simply fail to load and the script skips them.
- **Short prompts.** The test prompt is short, so the context (KV cache) is almost empty. In a long conversation that
  fills a 32K context, speed will be lower than measured, especially if part of the context sits in system RAM.
  Tune with the `--ctx` you really use, and re-check with long prompts if that matters to you.
- **One machine.** The table comes from one PC. It shows that tuning can matter a lot; it does not predict your numbers.
- `run_bench.bat` reproduces the table; the 9B and 14B rows use local model names (`unc9b`, `unc14b`) of
  community GGUF fine-tunes that you will not have — replace them with any 9B/14B model.

## License

MIT
