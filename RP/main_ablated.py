import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from scipy.sparse.linalg import expm_multiply
import scipy.sparse as sp

from eval import *
from dataset import load_dataset, dataset_statistics
from utils import *

# path for data/ folder containing datasets
from pathlib import Path
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent/"data"
data_dir = data_dir.as_posix()

class ExpABPropagation(nn.Module):
    '''
    Ablation: res(t+1) = exp(-eta * A * B) @ res(t)
    '''
    def __init__(self, eta):
        super().__init__()
        self.eta = eta
        self.AB = None

    def preprocess(self, args, edge_index, num_nodes, num_classes, train_mask):
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        
        
        A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD').coalesce()
        A_sp = sp.coo_matrix(
            (A.values().cpu().numpy(), (A.indices()[0].cpu().numpy(), A.indices()[1].cpu().numpy())),
            shape=A.shape
        )

        

        B_diag = train_mask.float().cpu().numpy()
        B = sp.diags(B_diag)

        
        
        AB = (A_sp @ B).tocsc()  # convert to CSC to avoid SparseEfficiencyWarning

        
        self.AB = -self.eta*AB
        

    def forward(self, residuals):
        r_cpu = residuals.cpu().numpy()
        r_next = expm_multiply(self.AB, r_cpu)
        return torch.tensor(r_next, device=residuals.device, dtype=residuals.dtype)
           
ALGORITHMS = {
    'LP': ExpABPropagation
}

def get_args():
    parser = argparse.ArgumentParser(prog='GNN Pipeline', description='Training pipeline for node classification')
    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--fixed_split', default=False, action='store_true', help='Use fixed_split for cora/citeseer/pubmed datasets')
    parser.add_argument('--train_ratio', type=float, default=None, help='Training set ratio for random split')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default = data_dir)

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--steps', type=int, default=100, help='number of steps for each run')
    parser.add_argument('--lr', type=float, default=0.01)

    # The following arguments are for RP-like algorithms
    parser.add_argument('--algorithm', type=str, default='LP')
    parser.add_argument('--alpha', type=float, default=1.0, help='hyperparameter alpha')
    parser.add_argument('--k', type=int, default=3, help='number of iterations')

    args = parser.parse_args()
    if args.name is None and args.algorithm is not None:
        args.name = args.algorithm
    return args

def main():
    
    args = get_args(); fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # Prepare dataset
    data = load_dataset(args.data_dir, args.dataset, args.runs, train_ratio=args.train_ratio, fixed_split=args.fixed_split)
    data.to(device)
    k, n, d, e = dataset_statistics(data, dataset_name = args.dataset, verbose = True)
    metric_name = data.metric

    logger = Logger(args, metric=metric_name)
    
    for run in range(1, args.runs + 1):
        

        run_idx = (run - 1) % data.train_mask.shape[1]
        train_mask, val_mask, test_mask = data.train_mask[:, run_idx], data.val_mask[:, run_idx], data.test_mask[:, run_idx]
        # Prepare model
        algorithm = ALGORITHMS[args.algorithm](eta=args.lr)
        algorithm.preprocess(args, data.edge_index, num_nodes=n, num_classes=k, train_mask=train_mask)
        algorithm.to(device)

        logger.start_run(run)

        # initialization
        residuals = torch.zeros_like(data.onehot_y)
        residuals[train_mask] = data.onehot_y[train_mask].clone()

        for step in range(1, args.steps + 1):
            # Training
            residuals = algorithm(residuals)

            # Evaluate
            train_metric = METRICS[metric_name](data.y[train_mask].unsqueeze(dim=-1), -residuals[train_mask] + data.onehot_y[train_mask]) * 100
            val_metric = METRICS[metric_name](data.y[val_mask].unsqueeze(dim=-1), - residuals[val_mask]) * 100
            test_metric = METRICS[metric_name](data.y[test_mask].unsqueeze(dim=-1), - residuals[test_mask]) * 100
            metrics = {
                f'train {metric_name}': train_metric,
                f'val {metric_name}': val_metric,
                f'test {metric_name}': test_metric
            }

            if step == 1:
                prev_train = None
                prev_val = None
                prev_test = None
            round_train = round(train_metric, 2)
            round_val = round(val_metric, 2)
            round_test = round(test_metric, 2)

            if (
                prev_train != round_train or
                prev_val != round_val or
                prev_test != round_test
            ):
                logger.update_metrics(metrics=metrics, step=step)
            prev_train = round_train
            prev_val = round_val
            prev_test = round_test

            

        logger.finish_run()
        
    logger.print_metrics_summary()
    logger.save_in_csv()

if __name__ == '__main__':
    main()