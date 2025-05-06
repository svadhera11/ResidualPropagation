import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from collections import defaultdict
import numpy as np

sns.set(style="whitegrid")

def hp_string(row):
    return f"alpha={row['alpha']}, k={row['k']}, lr={row['lr']}"

def best_of_chart(best_df, outdir):
    """Grouped bar: best_test by dataset × ablation (top result per)."""
    order_ds = sorted(best_df.dataset.unique())
    abls     = sorted(best_df.ablation.unique())

    # Use custom color schemes
    color_map = {
        'run': sns.color_palette('Blues', n_colors=len(abls)+2)[-3],
        'run1': sns.color_palette('Greens', n_colors=len(abls)+2)[-3],
        'run2': sns.color_palette('Reds', n_colors=len(abls)+2)[-3]
    }
    fallback_colors = sns.color_palette('tab10', len(abls))
    palette = {abl: color_map.get(abl, fallback_colors[i])
               for i, abl in enumerate(abls)}

    fig_width = max(8, 1.5 * len(order_ds))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.barplot(data=best_df, x='dataset', y='best_test',
                hue='ablation', ax=ax, palette=palette, order=order_ds)

    ax.set_ylim(0,100)
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Best Performance per Dataset (Top Val per Ablation)')

    # Rotate and wrap x-ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Legend with HPs
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for abl in labels:
        row = best_df.query('ablation == @abl').iloc[0]
        hp = hp_string(row)
        new_labels.append(f'{abl} ({hp})')

    ax.legend(handles, new_labels, title='Best Configs',
              bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    outpath = outdir/'best_of_all_datasets.png'
    fig.savefig(outpath, dpi=180); plt.close()
    return outpath

def param_sweep_charts(df, outdir):
    plots = []
    param_cols = ['alpha', 'k', 'lr']
    ablations = sorted(df.ablation.unique())

    for ds in sorted(df.dataset.unique()):
        for param in param_cols:
            if param not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6,4))
            subset = df[df.dataset == ds]
            sns.scatterplot(
                data=subset, x=param, y='best_val',
                hue='ablation', style='ablation', ax=ax
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel('Val Accuracy (%)')
            ax.set_title(f'{ds} – {param} sweep')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            fname = f'{ds}_{param}_sweep.png'
            fig.savefig(outdir/fname, dpi=150); plt.close()
            plots.append(outdir/fname)
    return plots

def main(args):
    df = pd.read_csv(args.summary)

    # Identify ablation group
    df['ablation'] = df.log_file.map(lambda s: re.split(r'\.', s, 1)[0])

    # Keep top 3 entries per (ablation, dataset)
    df = (df.sort_values('best_val', ascending=False)
            .groupby(['ablation', 'dataset'])
            .head(3).reset_index(drop=True))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot sweeps
    sweep_plots = param_sweep_charts(df, outdir)

    # Plot best performance across datasets
    best_rows = (df.sort_values('best_val', ascending=False)
                   .groupby(['ablation', 'dataset'])
                   .first().reset_index())
    best_plot = best_of_chart(best_rows, outdir)

    print(f'✅ Saved {len(sweep_plots)} sweep plots and 1 comparison plot to {outdir}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=str, default='summary.csv',
                        help='Path to the CSV summary file')
    parser.add_argument('--outdir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    main(args)

