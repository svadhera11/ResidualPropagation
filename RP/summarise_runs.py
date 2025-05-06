#!/usr/bin/env python3
"""
Parse run-log files produced by LP/main*.py experiments and
summarise best validation / test accuracies per hyper-parameter
setting & dataset.

Usage:
    python summarise_runs.py log1.txt log2.txt ...
    python summarise_runs.py *.txt --csv out.csv
"""
import argparse, re, csv, pathlib
from collections import defaultdict

# ───────────────────────── helpers ────────────────────────── #

# Regexes for the lines we need
RUN_HDR   = re.compile(r'^Running\s+(?P<ds>\S+)\s+\|\s+lr=(?P<lr>[0-9.]+)\s+\|\s+k=(?P<k>[0-9.]+)(.*)$')
FINISHED  = re.compile(r'Best val acc:\s+(?P<val>[0-9.]+),\s+corresponding test acc:\s+(?P<test>[0-9.]+)\s+\(step\s+(?P<step>\d+)')
HP_KV     = re.compile(r'(\w+)=([0-9.]+)')          # catches any extra “… | foo=bar” tokens

def parse_file(path: pathlib.Path):
    """Yield dicts with all fields for every (dataset, hp) block in *path*."""
    current = None
    with path.open() as f:
        for line in f:
            if m := RUN_HDR.match(line):
                # Flush the previous block if it never reached 'Finished run'
                if current is not None:
                    current['warn'] = 'no_finished_line'
                    yield current
                # start new block
                current = {
                    'dataset' : m['ds'],
                    'lr'      : float(m['lr']),
                    'k'       : int(float(m['k'])),   # k can appear as 1.0
                    'log_file': path.name
                }
                # Grab any extra “| foo=bar” hyper-params
                for hp, val in HP_KV.findall(line):
                    if hp not in ('lr', 'k'):
                        current[hp] = float(val)
            elif current and (m := FINISHED.search(line)):
                current.update({
                    'best_val' : float(m['val']),
                    'best_test': float(m['test']),
                    'best_step': int(m['step'])
                })
                yield current
                current = None
    # If the file ended without a Finished line
    if current is not None:
        current['warn'] = 'no_finished_line_at_eof'
        yield current

def nice_print(rows):
    """Pretty-print table to stdout."""
    if not rows:
        return
    cols = sorted({k for r in rows for k in r})
    # max width per column
    width = {c: max(len(c), *(len(str(r.get(c,''))) for r in rows)) for c in cols}
    fmt   = '  '.join('{:<%d}'%width[c] for c in cols)
    print(fmt.format(*cols))
    print('-'*(sum(width.values()) + 2*len(width)))
    for r in rows:
        print(fmt.format(*(r.get(c,'') for c in cols)))

# ─────────────────────────── main ─────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+', help='run_log files to parse')
    ap.add_argument('--csv', help='write the summary as CSV here')
    args = ap.parse_args()

    all_rows = []
    for p in map(pathlib.Path, args.logs):
        all_rows.extend(parse_file(p))

    # Sort for convenience: dataset → best_val desc
    all_rows.sort(key=lambda r: (r['dataset'], -r.get('best_val',0)))

    nice_print(all_rows)

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in all_rows for k in r}))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'\n✅  CSV written to: {args.csv}')

if __name__ == '__main__':
    main()

