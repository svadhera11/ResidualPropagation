#!/bin/bash

LOGFILE="run_log.txt"
echo "Resumed run at $(date)" >> $LOGFILE

for kernel in cosine sigmoid; do
  for lr in 1.0 2.0 3.0; do
    for k in 0 1; do
      echo "Running amazon-ratings | kernel=$kernel | lr=$lr | k=$k | gamma=1.0" | tee -a $LOGFILE
      python main.py --dataset amazon-ratings --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 1.0 --steps 50000 --runs 10 --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done

echo "Resumed run ended at $(date)" >> $LOGFILE

