import os
import json
import numpy as np
import matplotlib.pyplot as plt

METRICS_FILE = "../data/metrics_test.json"
PLOTS_DIR = "../plots/cpp_vs_python/tests"

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(METRICS_FILE):
    print("Файл ../data/metrics_test.json не найден")
    exit(1)

with open(METRICS_FILE) as f:
    data = json.load(f)

if "python" not in data or "cpp" not in data:
    print("Нет данных для 'python' или 'cpp'")
    exit(1)

python_data = data["python"]
cpp_data = data["cpp"]

MODELS = ["rf", "svm", "mlp"]
x = np.arange(len(MODELS))
width = 0.35

def get_metric(data, model, key_path):
    if model not in data:
        return 0
    entry = data[model][0] if isinstance(data[model], list) else data[model]
    keys = key_path.split(".")
    val = entry
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return 0
    return float(val) if isinstance(val, (int, float)) else 0

metrics = [
    ("latency_ms.mean", "Среднее время (мс)"),
    ("latency_ms.p90", "P90 (мс)"),
    ("latency_ms.p99", "P99 (мс)"),
    ("requests_total", "Запросов"),
    ("attacks_detected", "Обнаружено атак")
]

for key, title in metrics:
    py_vals = [get_metric(python_data, m, key) for m in MODELS]
    cpp_vals = [get_metric(cpp_data, m, key) for m in MODELS]

    plt.figure(figsize=(7, 4.5))
    bars1 = plt.bar(x - width/2, py_vals, width, label="python", color="#4C72B0", edgecolor="black")
    bars2 = plt.bar(x + width/2, cpp_vals, width, label="cpp", color="#55A868", edgecolor="black")
    plt.title(title)
    plt.xlabel("Модель")
    plt.ylabel(title)
    plt.xticks(x, MODELS)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, val in zip(bars1, py_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(py_vals+cpp_vals)*0.01,
                 f"{val:.1f}" if isinstance(val, float) else str(int(val)),
                 ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, cpp_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(py_vals+cpp_vals)*0.01,
                 f"{val:.1f}" if isinstance(val, float) else str(int(val)),
                 ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    filename = f"{key.replace('.', '_')}_test.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График: {PLOTS_DIR}/{filename}")

print("Готово")
