#!/bin/bash
# Sequential, one job at a time on the whole machine (6 parallel jobs were ~5x slower each).
cd "$(dirname "$0")"
PY=ref/QNNKDMindSpore/.venv/bin/python
mkdir -p out_kd
for job in "0 binary" "0 cont" "1 binary" "2 binary"; do
  set -- $job
  $PY -u qnnkd_mnist15.py $1 $2 4 6 8 > out_kd/${2}_s$1.log 2>&1
done
