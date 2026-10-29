@echo off
rem Benchmark used for the README table. Log: results\bench.log, data: results\bench.json
cd /d %~dp0
set PYTHONIOENCODING=utf-8
py -c "import json,urllib.request as u;[u.urlopen(u.Request(h+'/api/generate',json.dumps({'model':m['name'],'keep_alive':0}).encode()),timeout=60).read() for h in ('http://127.0.0.1:11434','http://127.0.0.1:11435') for m in json.load(u.urlopen(h+'/api/ps',timeout=5))['models']]" 2>nul
(
py -u tune_layers.py llama3.2:3b --out results\bench.json
py -u tune_layers.py qwen3:4b-instruct --out results\bench.json
py -u tune_layers.py gemma3:4b --out results\bench.json
py -u tune_layers.py qwen3:4b-instruct --ctx 32768 --out results\bench.json
py -u tune_layers.py qwen2.5:7b-instruct --out results\bench.json
py -u tune_layers.py unc9b --host http://127.0.0.1:11435 --out results\bench.json
py -u tune_layers.py unc14b --host http://127.0.0.1:11435 --out results\bench.json
echo ALL DONE
) > results\bench.log 2>&1
