#!/bin/bash
# E4b on the office machine: full GRID video download, labels, training (3 seeds x 2 arch x 2 GT).
cd ~/tsync_exp
. venv/bin/activate
(cd grid && python fetch_full.py > ../logs/fetch_full.log 2>&1) || true
cd ~/tsync_exp
for s in $(seq 1 34); do [ $s -eq 21 ] && continue; python e4full_build.py s$s > logs/e4fb_s$s.log 2>&1 & while [ $(jobs -r | wc -l) -ge 14 ]; do sleep 5; done; done; wait
python e4full_build.py merge > logs/e4fb_merge.log 2>&1
python -u e4full_train.py > logs/e4full_train.log 2>&1
echo E4FULL_DONE >> logs/e4full_train.log
