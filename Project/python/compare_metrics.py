import os
import json
import numpy as np
import matplotlib.pyplot as plt

METRICS_FILE = "../data/metrics.json"
PLOTS_DIR = "../plots"
SUMMARY_FILE = "../data/metrics_summary.txt"

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(METRICS_FILE):
    print(f"Файл {METRICS_FILE} не найден")
    exit(1)

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if "python" not in data:
    print("Колонка 'python' отсутствует в metrics.json")
    exit(1)

python_data = data["python"]
models = sorted(python_data.keys())

train_entries = {}
inference_entries = {}

for model in models:
    entries = python_data[model]
    for e in entries:
        stage = e.get("stage")
        if stage == "inference":
            inference_entries[model] = e
        elif stage is None or stage == "train":
            train_entries[model] = e

if not train_entries:
    print("Нет записей обучения в колонке 'python'")
    exit(1)

print("Модели (python):", ", ".join(models))

metrics_list = [
    ("accuracy", "Accuracy"),
    ("f1_macro", "F1 Macro"),
    ("precision_macro", "Precision Macro"),
    ("recall_macro", "Recall Macro"),
    ("f1_weighted", "F1 Weighted"),
    ("roc_auc", "ROC-AUC"),
    ("training_time_sec", "Training Time (sec)")
]

summary_lines = []
summary_lines.append("=== Сводка метрик (обучение, python) ===")

for model in models:
    if model in train_entries:
        m = train_entries[model]["metrics"]
        acc = m.get("accuracy", 0)
        f1m = m.get("f1_macro", 0)
        train_time = train_entries[model].get("training_time_sec", 0)
        summary_lines.append(f"{model}: acc={acc:.4f}, f1_macro={f1m:.4f}, time={train_time:.1f}s")

if inference_entries:
    summary_lines.append("\n=== Инференс (python, последняя сессия) ===")
    for model in models:
        if model in inference_entries:
            inf = inference_entries[model]["inference_time_ms"]
            reqs = inference_entries[model]["requests_total"]
            attacks = inference_entries[model]["attacks_detected"]
            avg_ms = inf.get("mean", 0)
            summary_lines.append(f"{model}: запросов={reqs}, атак={attacks}, avg_inference={avg_ms:.2f} мс")

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Сводка сохранена: {SUMMARY_FILE}")

for metric_key, metric_name in metrics_list:
    values = []
    labels = []
    for model in models:
        if model in train_entries:
            val = train_entries[model].get("metrics", {}).get(metric_key, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
                labels.append(model)
    if not values:
        continue

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color="#4C72B0", edgecolor="black", width=0.6)
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
    plot_path = os.path.join(PLOTS_DIR, f"{metric_key}_train_python.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График: {plot_path}")

if inference_entries:
    values = []
    labels = []
    for model in models:
        if model in inference_entries:
            val = inference_entries[model]["inference_time_ms"].get("mean", None)
            if val is not None:
                values.append(val)
                labels.append(model)
    if values:
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values, color="#C44E52", edgecolor="black", width=0.6)
        plt.title("Среднее время инференса (мс, python)", fontsize=12, pad=15)
        plt.ylabel("Время (мс)", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.015,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, "inference_time_ms_python.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"График: {plot_path}")

print("Готово")
