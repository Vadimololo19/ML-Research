import os
import json
import numpy as np
import matplotlib.pyplot as plt

METRICS_FILE = "../data/metrics_test.json"
PLOTS_DIR = "../plots/inference_comparison"

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(METRICS_FILE):
    print("Файл ../data/metrics_test.json не найден")
    exit(1)

with open(METRICS_FILE) as f:
    data = json.load(f)

if "python" not in data or "cpp" not in 
    print("Нет данных для 'python' или 'cpp'")
    exit(1)

def get_last_entry(lang, model):
    entries = lang.get(model, [])
    return entries[-1] if entries else None

models = sorted(set(data["python"].keys()) & set(data["cpp"].keys()))
if not models:
    print("Нет общих моделей")
    exit(1)

metrics = [
    ("latency_ms.mean", "Среднее время (мс)"),
    ("latency_ms.p90", "P90 (мс)"),
    ("latency_ms.p99", "P99 (мс)"),
    ("requests_total", "Запросов"),
    ("attacks_detected", "Обнаружено атак")
]

x = np.arange(len(models))
width = 0.35

for key_path, title in metrics:
    py_vals = []
    cpp_vals = []
    for model in models:
        py = get_last_entry(data["python"], model)
        cp = get_last_entry(data["cpp"], model)
        if not py or not cp:
            py_vals.append(0); cpp_vals.append(0); continue
        keys = key_path.split(".")
        py_val = py
        cp_val = cp
        for k in keys:
            py_val = py_val.get(k, 0)
            cp_val = cp_val.get(k, 0)
        py_vals.append(py_val)
        cpp_vals.append(cp_val)

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width/2, py_vals, width, label="python", color="#4C72B0", edgecolor="black")
    plt.bar(x + width/2, cpp_vals, width, label="cpp", color="#55A868", edgecolor="black")
    plt.title(title)
    plt.xlabel("Модель")
    plt.ylabel(title)
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    for i, (pyv, cpv) in enumerate(zip(py_vals, cpp_vals)):
        plt.text(i - width/2, pyv + max(py_vals)*0.01, f"{pyv:.1f}" if isinstance(pyv, float) else str(pyv), ha="center", va="bottom", fontsize=8)
        plt.text(i + width/2, cpv + max(py_vals+cpp_vals)*0.01, f"{cpv:.1f}" if isinstance(cpv, float) else str(cpv), ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{key_path.replace('.', '_')}_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График: {PLOTS_DIR}/{key_path.replace('.', '_')}_test.png")

print("Готово")
