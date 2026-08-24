#!/usr/bin/env bash
# GS learning-rate sweep. Runs from the repo root regardless of where invoked
# (cd to models/.. = K-Radar), and recreates the logs dir it writes to.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1   # models/.. -> repo root
mkdir -p models/logs

for i in 1e-3 5e-4 1e-4
do
    python models/GS.py --resolution 32,32,32 --lr "$i" >> models/logs/${i}.txt
done
