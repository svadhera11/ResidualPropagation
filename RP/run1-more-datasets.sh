#!/bin/bash

LOGFILE="run_log_1_moredata.txt"
echo "Run started at $(date)" > $LOGFILE

for lr in 0.01 0.02 0.05 0.1 0.2 0.5 1.0; do
  for k in 1 2 3 4 5 6 7 8 9 10; do

    echo "Running cs | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset cs --algorithm RK4 --lr $lr --alpha 1.0 --k $k --steps 5000 --runs 1 2>&1 | tee -a $LOGFILE  

    echo "Running physics | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset physics --algorithm RK4 --lr $lr --alpha 1.0 --k $k --steps 5000 --runs 1  2>&1 | tee -a $LOGFILE
    
    echo "Running questions | lr=$lr | k=$k" | tee -a $LOGFILE
    python main1.py --dataset questions --algorithm RK4 --lr $k --alpha 1.0 --k $k --steps 10000 --runs 10  2>&1 | tee -a $LOGFILE

  done
done

echo "Run ended at $(date)" >> $LOGFILE
