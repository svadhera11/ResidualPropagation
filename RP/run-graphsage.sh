LOGFILE="run_log_sagegraph.txt"
echo "Run started at $(date)" > $LOGFILE


# First, let's run on standard benchmark datasets with reasonable default parameters
echo "Running default configurations on benchmark datasets" | tee -a $LOGFILE

python sage.py --dataset cora --fixed_split  --lr 0.005  --device cpu --runs 2 2>&1 | tee -a $LOGFILE

echo "Running default configurations on benchmark datasets" | tee -a $LOGFILE
python sage.py --dataset citeseer --fixed_split  --lr 0.005 --device cpu --runs 2 2>&1 | tee -a $LOGFILE

echo "Running default configurations on benchmark datasets" | tee -a $LOGFILE
python sage.py --dataset pubmed --fixed_split  --lr 0.005 --device cpu --runs 2 2>&1 | tee -a $LOGFILE

echo "Running default configurations on benchmark datasets" | tee -a $LOGFILE
python sage.py --dataset ogbn-arxiv --fixed_split  --lr 0.005 --device cpu --runs 2 2>&1 | tee -a $LOGFILE

echo "Running default configurations on benchmark datasets" | tee -a $LOGFILE
python sage.py --dataset cs --fixed_split  --lr 0.005 --device cpu --runs 2 2>&1 | tee -a $LOGFILE

echo "Running default configurations on benchmark datasets" | tee -a $LOGFILE
python sage.py --dataset physics --fixed_split  --lr 0.005 --device cpu --runs 2 2>&1 | tee -a $LOGFILE
