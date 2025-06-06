#!/bin/bash

LOGFILE="run_log.txt"
echo "Run started at $(date)" > $LOGFILE

for kernel in gaussian cosine sigmoid; do
  for lr in 5.0 10.0 15.0; do
    for k in 2 3 4; do
      echo "Running cora | kernel=$kernel | lr=$lr | k=$k | gamma=1.0" | tee -a $LOGFILE
      python main.py --dataset cora --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 1.0 --steps 20000 --fixed_split --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done

for kernel in gaussian cosine sigmoid; do
  for lr in 5.0 10.0 15.0; do
    for k in 2 3 4; do
      echo "Running citeseer | kernel=$kernel | lr=$lr | k=$k | gamma=20.0" | tee -a $LOGFILE
      python main.py --dataset citeseer --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 20.0 --steps 1000 --fixed_split --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done

for kernel in gaussian cosine sigmoid; do
  for lr in 50 75 100 125; do
    for k in 3 4 5; do
      echo "Running pubmed | kernel=$kernel | lr=$lr | k=$k | gamma=1.0" | tee -a $LOGFILE
      python main.py --dataset pubmed --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 1.0 --steps 50000 --fixed_split --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done

for kernel in gaussian cosine sigmoid; do
  for lr in 0.5 1.0 2.0; do
    for k in 0 1; do
      echo "Running roman-empire | kernel=$kernel | lr=$lr | k=$k | gamma=1.0" | tee -a $LOGFILE
      python main.py --dataset roman-empire --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 1.0 --steps 10000 --runs 10 --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done

for kernel in gaussian cosine sigmoid; do
  for lr in 1.0 2.0 3.0; do
    for k in 0 1; do
      echo "Running amazon-ratings | kernel=$kernel | lr=$lr | k=$k | gamma=1.0" | tee -a $LOGFILE
      python main.py --dataset amazon-ratings --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 1.0 --steps 50000 --runs 10 --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done


for kernel in gaussian cosine sigmoid; do
  for lr in 0.5 1.0 2.0; do
    for k in 1 2 3; do
      echo "Running cs | kernel=$kernel | lr=$lr | k=$k | gamma=1.0" | tee -a $LOGFILE
      python main.py --dataset cs --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 20 --steps 5000 --runs 1 --device cuda:1 2>&1 | tee -a $LOGFILE    
    done
  done
done

for kernel in gaussian cosine sigmoid; do
  for lr in 0.5 1.0 2.0; do
    for k in 0 1 2; do
      echo "Running citeseer | kernel=$kernel | lr=$lr | k=$k | gamma=20.0" | tee -a $LOGFILE
      python main.py --dataset physics --algorithm LP --kernel $kernel --lr $lr --alpha 1.0 --k $k --gamma 20 --steps 5000 --runs 1 --device cuda:1 2>&1 | tee -a $LOGFILE
    done
  done
done


echo "Run ended at $(date)" >> $LOGFILE


#---
#python main.py --dataset cora --algorithm LP --lr 10 --alpha 1.0 --k 3 --gamma 1 --steps 20000 --fixed_split 

#python main.py --dataset citeseer --algorithm LP --lr 10 --alpha 1.0 --k 3 --gamma 20 --steps 1000 --fixed_split 

#python main.py --dataset pubmed --algorithm LP --lr 100 --alpha 1.0 --k 4 --gamma 1 --steps 50000 --fixed_split 

#python main.py --dataset computers --algorithm LP --lr 1 --alpha 1.0 --k 1 --gamma 0.25 --steps 100 --runs 1

#python main.py --dataset photo --algorithm LP --lr 1 --alpha 1.0 --k 3 --gamma 0.1 --steps 500 --runs 1

#python main.py --dataset cs --algorithm LP --lr 1 --alpha 1.0 --k 1 --gamma 20 -2>&1-steps 5000 --runs 1

#python main.py --dataset physics --algorithm LP --lr 1 --alpha 1.0 --k 1 --gamma 20 --steps 5000 --runs 1

#python main.py --dataset roman-empire --algorithm LP --lr 1 --alpha 1.0 --k 0 --gamma 1 --steps 10000 --runs 10

#python main.py --dataset amazon-ratings --algorithm LP --lr 2 --alpha 1.0 --k 0 --gamma 1 --steps 50000 --runs 10

#python main.py --dataset minesweeper --algorithm LP --lr 1 --alpha 1 --k 1 --gamma 10 --steps 1000 --runs 10

#python main.py --dataset tolokers --algorithm LP --lr 1 --alpha 0.5 --k 3 --gamma 10 --steps 1000 --runs 10

#python main.py --dataset questions --algorithm LP --lr 5 --alpha 1.0 --k 2 --gamma 1 --steps 10000 --runs 10
