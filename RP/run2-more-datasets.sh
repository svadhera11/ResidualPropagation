#!/bin/bash

LOGFILE="run_log_2_moredata.txt"
echo "Run started at $(date)" > $LOGFILE

for lr in 0.01 0.05 0.1 0.5 1.0 10.0; do
  for k in 1 2 3 4; do
    for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
      echo "Running cs | lr=$lr | k=$k" | tee -a $LOGFILE
      python main2.py --dataset cs --algorithm LP --lr $lr --alpha 1.0 --k $k --steps 5000 --runs 1 2>&1 | tee -a $LOGFILE  

      echo "Running physics | lr=$lr | k=$k" | tee -a $LOGFILE
      python main2.py --dataset physics --algorithm LP --lr $lr --alpha 1.0 --k $k --steps 5000 --runs 1  2>&1 | tee -a $LOGFILE
      
      echo "Running questions | lr=$lr | k=$k" | tee -a $LOGFILE
      python main2.py --dataset questions --algorithm LP --lr $k --alpha 1.0 --k $k --steps 10000 --runs 10  2>&1 | tee -a $LOGFILE
    done
  done
done

echo "Run ended at $(date)" >> $LOGFILE
