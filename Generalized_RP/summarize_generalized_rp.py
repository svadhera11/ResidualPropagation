import re
from collections import defaultdict

log_file = "run_log.txt"

# Updated regex patterns
run_start_re = re.compile(r"Running (\S+) \| kernel=(\S+) \| lr=([\d.]+) \| k=(\d+) \| gamma=([\d.]+)")
step_re = re.compile(r"step: *(\d+), train acc: *([\d.]+), val acc: *([\d.]+), test acc: *([\d.]+)")
summary_re = re.compile(r"Best val acc: ([\d.]+), corresponding test acc: ([\d.]+) \(step (\d+)\)")

# Store results per (dataset, kernel)
runs_by_group = defaultdict(list)
current = None

with open(log_file, "r") as f:
    for line in f:
        start = run_start_re.search(line)
        step = step_re.search(line)
        summary = summary_re.search(line)

        if start:
            if current:
                key = (current["dataset"], current["kernel"])
                runs_by_group[key].append(current)

            dataset, kernel, lr, k, gamma = start.groups()
            current = {
                "dataset": dataset,
                "kernel": kernel,
                "lr": float(lr),
                "k": int(k),
                "gamma": float(gamma),
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
        key = (current["dataset"], current["kernel"])
        runs_by_group[key].append(current)

# === Print best results per dataset+kernel ===
print("\n=== Best Runs Per Dataset + Kernel ===\n")

for (dataset, kernel), runs in sorted(runs_by_group.items()):
    best = None
    for run in runs:
        if run["summary"]:
            if best is None or run["summary"]["best_val"] > best["summary"]["best_val"]:
                best = run

    if best:
        s = best["summary"]
        print(f"📊 {dataset} | kernel={kernel}")
        print(f"  Best val acc: {s['best_val']:.2f} at step {s['step']}")
        print(f"  Corresponding test acc: {s['best_test']:.2f}")
        print(f"  Params: lr={best['lr']}, k={best['k']}, gamma={best['gamma']}")
        print("-" * 40)