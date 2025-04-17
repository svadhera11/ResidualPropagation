import re
import matplotlib.pyplot as plt
from collections import defaultdict

log_file = "run_log_2.txt"

# Patterns
run_start_re = re.compile(r"Running (.*?) \| lr=([\d.]+) \| k=(\d+)")
step_re = re.compile(r"step: *(\d+), train acc: *([\d.]+), val acc: *([\d.]+), test acc: *([\d.]+)")
summary_re = re.compile(r"Best val acc: ([\d.]+), corresponding test acc: ([\d.]+) \(step (\d+)\)")

# Storage
runs_by_dataset = defaultdict(list)
current = None

with open(log_file, "r") as f:
    for line in f:
        start = run_start_re.search(line)
        step = step_re.search(line)
        summary = summary_re.search(line)

        if start:
            if current:
                runs_by_dataset[current["dataset"]].append(current)
            dataset, lr, k = start.groups()
            current = {
                "dataset": dataset,
                "lr": float(lr),
                "k": int(k),
                "steps": [],
                "val_accs": [],
                "test_accs": [],
                "summary": None
            }

        elif step and current:
            s, train, val, test = step.groups()
            current["steps"].append(int(s))
            current["val_accs"].append(float(val))
            current["test_accs"].append(float(test))

        elif summary and current:
            val, test, step_val = summary.groups()
            current["summary"] = {
                "best_val": float(val),
                "best_test": float(test),
                "step": int(step_val)
            }

    if current:
        runs_by_dataset[current["dataset"]].append(current)

# Summary for all datasets
print("\n=== Best Runs Per Dataset ===\n")
for dataset, runs in runs_by_dataset.items():
    best = None

    plt.figure(figsize=(8, 4))
    for run in runs:
        if not run["val_accs"]:
            continue

        label = f'lr={run["lr"]}, k={run["k"]}'
        plt.plot(run["steps"], run["val_accs"], label=f'Val ({label})')
        plt.plot(run["steps"], run["test_accs"], linestyle='--', label=f'Test ({label})')

        if run["summary"]:
            if best is None or run["summary"]["best_val"] > best["summary"]["best_val"]:
                best = run

    plt.title(f"{dataset} - Validation/Test Accuracy Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.close()

    if best and best["summary"]:
        s = best["summary"]
        print(f"📊 {dataset}")
        print(f"  Best val acc: {s['best_val']:.2f} at step {s['step']}")
        print(f"  Corresponding test acc: {s['best_test']:.2f}")
        print(f"  Params: lr={best['lr']}, k={best['k']}")
        print("-" * 40)
