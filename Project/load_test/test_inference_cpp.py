import time
import json
import requests
import sys
import os
import numpy as np
import threading

MODELS = ["rf", "svm", "mlp"]
PORTS = [6000, 6001, 6002]
TOTAL_REQUESTS = 200

with open("payloads/safe.json") as f:
    safe_payloads = json.load(f)
with open("payloads/attack.json") as f:
    attack_payloads = json.load(f)

results = {}

def send_batch(model, port):
    url = f"http://127.0.0.1:{port}/predict"
    latencies = []
    requests_ok = 0
    attacks_detected = 0
    errors = 0
    
    for i in range(TOTAL_REQUESTS):
        is_attack = (i % 7 == 0)
        payload = attack_payloads[i % len(attack_payloads)] if is_attack else safe_payloads[i % len(safe_payloads)]
        start = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=5)
            lat = (time.perf_counter() - start) * 1000
            if r.status_code == 200:
                data = r.json()
                latencies.append(lat)
                requests_ok += 1
                if data.get("is_attack"):
                    attacks_detected += 1
            else:
                errors += 1
        except:
            errors += 1
        time.sleep(0.001)
    
    results[model] = {
        "requests_total": requests_ok,
        "attacks_detected": attacks_detected,
        "errors": errors,
        "latencies": latencies
    }

def run_tests():
    threads = []
    for i, model in enumerate(MODELS):
        port = PORTS[i]
        t = threading.Thread(target=send_batch, args=(model, port))
        threads.append(t)
        t.start()
        time.sleep(0.2)
    for t in threads:
        t.join()

def save_metrics():
    metrics_path = "../data/metrics_test.json"
    all_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                all_data = json.load(f)
        except:
            pass
    
    if "cpp" not in all_data:
        all_data["cpp"] = {}
    
    for model in MODELS:
        if model not in results:
            continue
        res = results[model]
        lat = res["latencies"]
        entry = {
            "timestamp": time.time(),
            "model": model,
            "requests_total": res["requests_total"],
            "attacks_detected": res["attacks_detected"],
            "errors": res["errors"],
            "latency_ms": {
                "mean": float(np.mean(lat)) if lat else 0,
                "p50": float(np.percentile(lat, 50)) if lat else 0,
                "p90": float(np.percentile(lat, 90)) if lat else 0,
                "p99": float(np.percentile(lat, 99)) if lat else 0,
                "max": float(np.max(lat)) if lat else 0
            }
        }
        all_data["cpp"][model] = [entry]  # перезапись
    
    os.makedirs("../data", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_data, f, indent=2)

if __name__ == "__main__":
    print("Запуск нагрузочного теста C++...")
    run_tests()
    save_metrics()
    print("Готово. Результаты перезаписаны в ../data/metrics_test.json")
