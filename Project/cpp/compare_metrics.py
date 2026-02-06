import os
import json
import numpy as np
import matplotlib.pyplot as plt

METRICS_FILE = "../data/metrics.json"
PLOTS_DIR = "../plots"
SUMMARY_FILE = "../data/metrics_summary_cpp.txt"

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(METRICS_FILE):
    print("Файл ../data/metrics.json не найден")
    exit(1)

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if "cpp" not in data:
    print("Колонка 'cpp' отсутствует")
    exit(1)

cpp_data = data["cpp"]
models = sorted(cpp_data.keys())

entries = {}
for model in models:
    entries[model] = cpp_data[model][-1]

if not entries:
    print("Нет записей в 'cpp'")
    exit(1)

print("Модели (cpp):", ", ".join(models))

metrics_list = [
    ("accuracy", "Accuracy"),
    ("f1_macro", "F1 Macro"),
    ("precision_macro", "Precision Macro"),
    ("recall_macro", "Recall Macro"),
    ("training_time_sec", "Training Time (sec)")
]

summary_lines = []
summary_lines.append("=== Сводка метрик (cpp) ===")

for model in models:
    e = entries[model]
    m = e["metrics"]
    acc = m.get("accuracy", 0)
    f1 = m.get("f1_macro", 0)
    t = e.get("training_time_sec", 0)
    summary_lines.append(f"{model}: acc={acc:.4f}, f1={f1:.4f}, time={t:.1f}s")

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Сводка сохранена: {SUMMARY_FILE}")

for metric_key, metric_name in metrics_list:
    values = []
    labels = []
    for model in models:
        if model in entries:
            if metric_key == "training_time_sec":
                val = entries[model].get(metric_key, 0)
            else:
                val = entries[model]["metrics"].get(metric_key, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
                labels.append(model)
    if not values:
        continue

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color="#55A868", edgecolor="black", width=0.6)
    plt.title(metric_name, fontsize=12, pad=15)
    plt.ylabel(metric_name, fontsize=10)
    if "time" not in metric_key.lower():
        plt.ylim(0, min(1.05, max(values) * 1.1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, val in zip(bars, values):
        if "time" in metric_key.lower():
            label = f"{val:.1f}"
        else:
            label = f"{val:.3f}"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (plt.ylim()[1] - plt.ylim()[0]) * 0.015,
                 label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{metric_key}_cpp.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График: {plot_path}")

print("Готово")
