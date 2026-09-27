#!/bin/bash
cd /home/wpu/tsync_exp
OUT=logs/c2_all.jsonl; : > $OUT
E1=atlas/paper/tools/layer1_audio
run(){ python3 layer1_wall.py "$@" 2>/dev/null >> $OUT; }
for lg in en tr; do
  if [ $lg = en ]; then F=$(ls fleurs/en_us/*.wav | head -20); else F=$(ls fleurs/tr_tr/*.wav | head -20); fi
  for m in tiny base; do
    run --paper-clock --model $m --lang $lg --tag espeak --wavs $E1/${lg}*.wav
    run --model $m --lang $lg --tag espeak --wavs $E1/${lg}*.wav
    run --paper-clock --model $m --lang $lg --tag fleurs20 --wavs $F
    run --model $m --lang $lg --tag fleurs20 --wavs $F
    run --grow --model $m --lang $lg --tag fleurs20 --wavs $F
  done
done
F=$(ls fleurs/en_us/*.wav | head -20)
run --paper-clock --model small --lang en --tag fleurs20 --wavs $F
run --model small --lang en --tag fleurs20 --wavs $F
run --grow --model small --lang en --tag fleurs20 --wavs $F
echo C2DONE >> logs/c2_done.txt
