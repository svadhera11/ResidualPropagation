#!/bin/bash

LOGFILE="run_log_2.txt"
echo "Run started at $(date)" > $LOGFILE

for lr in 0.01 0.05 0.1 0.5 1.0 10.0; do
  for k in 1 2 3 4; do
    for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        echo "Running ogbn-arxiv | lr=$lr | k=$k | alpha=$alpha" | tee -a $LOGFILE
        python main2.py --dataset ogbn-arxiv --algorithm LP --lr $lr --k $k --alpha $alpha --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

        echo "Running cora | lr=$lr | k=$k | alpha=$alpha" | tee -a $LOGFILE
        python main2.py --dataset cora --algorithm LP --fixed_split --lr $lr --k $k --alpha $alpha --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

        echo "Running citeseer | lr=$lr | k=$k | alpha=$alpha" | tee -a $LOGFILE
        python main2.py --dataset citeseer --algorithm LP --fixed_split --lr $lr --k $k --alpha $alpha --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

        echo "Running pubmed | lr=$lr | k=$k | alpha=$alpha" | tee -a $LOGFILE
        python main2.py --dataset pubmed --algorithm LP --fixed_split --lr $lr --k $k --alpha $alpha --steps 100 --runs 1 2>&1 | tee -a $LOGFILE
    done
  done
done

echo "Run ended at $(date)" >> $LOGFILE
