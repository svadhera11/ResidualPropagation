#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Node‑classification training pipeline with GraphSAGE.
# Everything except the model itself is identical to the original script.

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T

# ------------- project‑local helpers (unchanged) -----------------------------
from eval import *                        # METRICS dict lives here
from dataset import load_dataset, dataset_statistics
from utils import *                       # Logger, fix_seed, etc.
# -----------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                               Model definition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class GraphSageNet(nn.Module):
    """2‑ or 3‑layer GraphSAGE with ReLU + dropout."""
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        assert num_layers >= 2, "num_layers must be ≥ 2"

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    # For API‑compatibility with the old script (which called .preprocess)
    def preprocess(self, *args, **kwargs):        # noqa: D401
        """No preprocessing needed for GraphSAGE."""
        return

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 CLI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_args():
    parser = argparse.ArgumentParser(
        prog='GNN Pipeline',
        description='Training pipeline for node classification (GraphSAGE).'
    )

    # ─── dataset & run control ───────────────────────────────────────────────
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--fixed_split', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default=(Path(__file__).resolve().parent.parent / "data").as_posix())

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=600)          # epochs
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # ─── model hyper‑params ──────────────────────────────────────────────────
    parser.add_argument('--algorithm', type=str, default='SAGE',
                        choices=['SAGE'])                           # only SAGE in this script
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    if args.name is None:
        args.name = args.algorithm
    return args


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    args = get_args()
    fix_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ─── Load data ──────────────────────────────────────────────────────────
    data = load_dataset(args.data_dir,
                        args.dataset,
                        args.runs,
                        train_ratio=args.train_ratio,
                        fixed_split=args.fixed_split)
    data = T.ToDevice(device)(data)              # feature, edge_index, masks to GPU
    num_classes, num_nodes, num_feats, _ = dataset_statistics(
        data, dataset_name=args.dataset, verbose=True
    )
    metric_name = data.metric                    # 'acc' on Cora/Citeseer/Pubmed ...

    logger = Logger(args, metric=metric_name)

    # ─── Loop over independent runs ─────────────────────────────────────────
    for run in range(1, args.runs + 1):
        run_idx = (run - 1) % data.train_mask.shape[1]
        train_mask = data.train_mask[:, run_idx]
        val_mask   = data.val_mask[:, run_idx]
        test_mask  = data.test_mask[:, run_idx]

        # ── Instantiate model ───────────────────────────────────────────────
        model = GraphSageNet(
            in_channels=num_feats,
            hidden_channels=args.hidden_dim,
            out_channels=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        model.preprocess()                      # no‑op (keeps old API happy)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        logger.start_run(run)

        # ── Training loop ───────────────────────────────────────────────────
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            # ── Evaluation ────────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)

                train_metric = METRICS[metric_name](
                    data.y[train_mask].unsqueeze(-1),
                    logits[train_mask]
                ) * 100.0
                val_metric = METRICS[metric_name](
                    data.y[val_mask].unsqueeze(-1),
                    logits[val_mask]
                ) * 100.0
                test_metric = METRICS[metric_name](
                    data.y[test_mask].unsqueeze(-1),
                    logits[test_mask]
                ) * 100.0

            logger.update_metrics({
                f'train {metric_name}': train_metric,
                f'val {metric_name}':   val_metric,
                f'test {metric_name}':  test_metric,
                'loss':                loss.item()
            }, step=step)

        logger.finish_run()

    # ─── Summary & CSV dump ────────────────────────────────────────────────
    logger.print_metrics_summary()
    logger.save_in_csv()


if __name__ == '__main__':
    main()
