import re
from collections import defaultdict

LOGFILE = "run_log_1.txt"

# Regex patterns
header_re = re.compile(r"Running (\S+) \| lr=(\S+) \| k=(\S+)")
result_re = re.compile(
    r"step: ([\d\.]+), train acc: ([\d\.]+), val acc: ([\d\.]+), test acc: ([\d\.]+)"
)

# Track best results
best_results = defaultdict(lambda: {"val": -1, "step": None, "lr": None, "k": None, "train": None, "test": None})

with open(LOGFILE, "r") as f:
    current_dataset, current_lr, current_k = None, None, None
    for line in f:
        header_match = header_re.search(line)
        if header_match:
            current_dataset, current_lr, current_k = header_match.groups()
            continue

        result_match = result_re.search(line)
        if result_match:
            step, train_acc, val_acc, test_acc = map(float, result_match.groups())
            key = current_dataset

            if val_acc > best_results[key]["val"]:
                best_results[key].update({
                    "val": val_acc,
                    "train": train_acc,
                    "test": test_acc,
                    "step": step,
                    "lr": float(current_lr),
                    "k": int(current_k)
                })

# Print results
print("\n=== Best Results Per Dataset ===")
for dataset, stats in best_results.items():
    print(f"\n📊 Dataset: {dataset}")
    print(f"🔧 Best val acc: {stats['val']:.2f}% at step {stats['step']}")
    print(f"🏋️‍♂️ Train acc: {stats['train']:.2f}% | 🧪 Test acc: {stats['test']:.2f}%")
    print(f"⚙️  lr={stats['lr']}, k={stats['k']}")
