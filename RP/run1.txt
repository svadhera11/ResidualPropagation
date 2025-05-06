#!/bin/bash

LOGFILE="run_log_1.txt"
echo "Run started at $(date)" > $LOGFILE

for lr in 0.01 0.02 0.05 0.1 0.2 0.5 1.0; do
  for k in 1 2 3 4 5 6 7 8 9 10; do

    echo "Running ogbn-arxiv | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset ogbn-arxiv --algorithm RK4 --lr $lr --k $k --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running cora | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset cora --algorithm RK4 --fixed_split --lr $lr --k $k --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running citeseer | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset citeseer --algorithm RK4 --fixed_split --lr $lr --k $k --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

    echo "Running pubmed | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset pubmed --algorithm RK4 --fixed_split --lr $lr --k $k --steps 100 --runs 1 2>&1 | tee -a $LOGFILE

  done
done

echo "Run ended at $(date)" >> $LOGFILE
