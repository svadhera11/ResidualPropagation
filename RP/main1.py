import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from eval import *
from dataset import load_dataset, dataset_statistics
from utils import *

# path for data/ folder containing datasets
from pathlib import Path
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent/"data"
data_dir = data_dir.as_posix()

class RK4Propagation(nn.Module):
    def __init__(self, k, eta, dt=0.2):
        super().__init__()
        self.k = k
        self.eta = eta
        self.dt = dt
        self.A = None
        self.train_mask = None

    def preprocess(self, args, edge_index, num_nodes, train_mask):
        self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD').to(args.device)
        self.train_mask = train_mask.to(args.device)

    def apply_Ak_B(self, r):
        # Apply B (mask out non-train nodes)
        r = r * self.train_mask.unsqueeze(-1)
        # Apply A k times
        for _ in range(self.k):
            r = torch.sparse.mm(self.A, r)
        return -self.eta * r

    def propagate(self, r0):
        steps = int(1.0 / self.dt)
        out = [r0]
        r = r0
        for _ in range(steps):
            k1 = self.apply_Ak_B(r)
            k2 = self.apply_Ak_B(r + 0.5 * self.dt * k1)
            k3 = self.apply_Ak_B(r + 0.5 * self.dt * k2)
            k4 = self.apply_Ak_B(r + self.dt * k3)
            r = r + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            out.append(r)
        return out



ALGORITHMS = {
    'RK4': RK4Propagation
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
    parser.add_argument('--dt', type=float, default=0.2, help='RK4 timestep size')


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
        algorithm = ALGORITHMS[args.algorithm](k=args.k,
                                                eta=args.lr, dt=args.dt)
        algorithm.preprocess(args, data.edge_index, num_nodes=n, train_mask=train_mask)
        algorithm.to(device)


        logger.start_run(run)

        # initialization
        redisuals = torch.zeros_like(data.onehot_y)
        redisuals[train_mask] = data.onehot_y[train_mask].clone()

        for step in range(1, args.steps + 1):
            # Training
            masked_residuals = redisuals.clone()
            masked_residuals[~train_mask] =  0
            states = algorithm.propagate(redisuals)
            for i, r_t in enumerate(states):
                t_global = (step - 1) + i * args.dt
                r_masked = r_t.clone()
                r_masked[~train_mask] = 0

                train_metric = METRICS[metric_name](data.y[train_mask].unsqueeze(-1), -r_t[train_mask] + data.onehot_y[train_mask]) * 100
                val_metric = METRICS[metric_name](data.y[val_mask].unsqueeze(-1), -r_t[val_mask]) * 100
                test_metric = METRICS[metric_name](data.y[test_mask].unsqueeze(-1), -r_t[test_mask]) * 100

                metrics = {
                    f'train {metric_name}': train_metric,
                    f'val {metric_name}': val_metric,
                    f'test {metric_name}': test_metric
                }

                round_train = round(train_metric, 2)
                round_val = round(val_metric, 2)
                round_test = round(test_metric, 2)

                if step == 1 or (
                    round_train != prev_train or
                    round_val != prev_val or
                    round_test != prev_test
                ):
                    logger.update_metrics(metrics=metrics, step=f"{t_global:.2f}")

                prev_train = round_train
                prev_val = round_val
                prev_test = round_test

            # update residuals to the last state
            redisuals = states[-1]
        logger.finish_run()
        
    logger.print_metrics_summary()
    logger.save_in_csv()

if __name__ == '__main__':
    main()