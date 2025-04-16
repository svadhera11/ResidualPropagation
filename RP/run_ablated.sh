#!/bin/bash

LOGFILE="run_log_ablated.txt"
echo "Run started at $(date)" > $LOGFILE

for lr in 0.01 0.02 0.05 0.1 0.2 0.5 1.0; do
  for k in 1 2 3 4 5 6 7 8 9 10; do

    echo "Running ogbn-arxiv | lr=$lr | k=$k" | tee -a $LOGFILE
    python main_ablated.py --dataset ogbn-arxiv --algorithm LP --lr $lr --k $k --steps 1000 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running cora | lr=$lr | k=$k" | tee -a $LOGFILE
    python main_ablated.py --dataset cora --algorithm LP --fixed_split --lr $lr --k $k --steps 1000 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running citeseer | lr=$lr | k=$k" | tee -a $LOGFILE
    python main_ablated.py --dataset citeseer --algorithm LP --fixed_split --lr $lr --k $k --steps 1000 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running pubmed | lr=$lr | k=$k" | tee -a $LOGFILE
    python main_ablated.py --dataset pubmed --algorithm LP --fixed_split --lr $lr --k $k --steps 1000 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running roman-empire | lr=$lr | k=$k" | tee -a $LOGFILE
    python main_ablated.py --dataset roman-empire --algorithm LP --lr $lr --k $k --steps 1000 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running amazon-ratings | lr=$lr | k=$k" | tee -a $LOGFILE
    python main_ablated.py --dataset amazon-ratings --algorithm LP --lr $lr --k $k --steps 1000 --runs 1 2>&1 | tee -a $LOGFILE

  done
done

echo "Run ended at $(date)" >> $LOGFILE
