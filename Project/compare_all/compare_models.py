import os
import json
import numpy as np
import matplotlib.pyplot as plt

METRICS_FILE = "../data/metrics.json"
PLOTS_DIR = "../plots/cpp_vs_python"
SUMMARY_FILE = "../data/comparison_summary.txt"

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(METRICS_FILE):
    print("Файл ../data/metrics.json не найден")
    exit(1)

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if "python" not in data or "cpp" not in data:
    print("Нет данных для 'python' или 'cpp'")
    exit(1)

python_data = data["python"]
cpp_data = data["cpp"]

common_models = sorted(set(python_data.keys()) & set(cpp_data.keys()))
if not common_models:
    print("Нет общих моделей между 'python' и 'cpp'")
    exit(1)

print("Общие модели:", ", ".join(common_models))

def get_last_entry(lang_data, model):
    entries = lang_data[model]
    train_entries = [e for e in entries if e.get("stage") in (None, "train")]
    return train_entries[-1] if train_entries else None

metrics_list = [
    ("accuracy", "Accuracy"),
    ("f1_macro", "F1 Macro"),
    ("precision_macro", "Precision Macro"),
    ("recall_macro", "Recall Macro"),
    ("training_time_sec", "Training Time (sec)")
]

summary_lines = []
summary_lines.append("=== Сравнение cpp vs python (последние записи) ===")

for model in common_models:
    py = get_last_entry(python_data, model)
    cp = get_last_entry(cpp_data, model)
    if not py or not cp:
        continue
    py_m = py["metrics"]
    cp_m = cp["metrics"]
    py_acc = py_m.get("accuracy", 0)
    cp_acc = cp_m.get("accuracy", 0)
    py_f1 = py_m.get("f1_macro", 0)
    cp_f1 = cp_m.get("f1_macro", 0)
    py_t = py.get("training_time_sec", 0)
    cp_t = cp.get("training_time_sec", 0)
    summary_lines.append(f"{model}: py(acc={py_acc:.4f}, f1={py_f1:.4f}, t={py_t:.1f}s) | cpp(acc={cp_acc:.4f}, f1={cp_f1:.4f}, t={cp_t:.1f}s)")

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Сводка сохранена: {SUMMARY_FILE}")

x = np.arange(len(common_models))
width = 0.35

for metric_key, metric_name in metrics_list:
    py_vals = []
    cpp_vals = []
    labels = []
    for model in common_models:
        py = get_last_entry(python_data, model)
        cp = get_last_entry(cpp_data, model)
        if not py or not cp:
            continue
        if metric_key == "training_time_sec":
            py_val = py.get(metric_key, 0)
            cpp_val = cp.get(metric_key, 0)
        else:
            py_val = py["metrics"].get(metric_key, 0)
            cpp_val = cp["metrics"].get(metric_key, 0)
        if py_val is None or cpp_val is None:
            continue
        py_vals.append(py_val)
        cpp_vals.append(cpp_val)
        labels.append(model)

    if not py_vals:
        continue

    plt.figure(figsize=(7, 4.5))
    bars1 = plt.bar(x - width/2, py_vals, width, label="python", color="#4C72B0", edgecolor="black")
    bars2 = plt.bar(x + width/2, cpp_vals, width, label="cpp", color="#55A868", edgecolor="black")
    plt.xlabel("Модель")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name}: cpp vs python")
    plt.xticks(x, labels)
    plt.legend()
    if "time" not in metric_key.lower():
        plt.ylim(0, min(1.05, max(py_vals + cpp_vals) * 1.1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, val in zip(bars1, py_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (plt.ylim()[1] - plt.ylim()[0]) * 0.01,
                 f"{val:.3f}" if "time" not in metric_key else f"{val:.1f}",
                 ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, cpp_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (plt.ylim()[1] - plt.ylim()[0]) * 0.01,
                 f"{val:.3f}" if "time" not in metric_key else f"{val:.1f}",
                 ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{metric_key}_cpp_vs_python.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График: {plot_path}")

print("Готово")
