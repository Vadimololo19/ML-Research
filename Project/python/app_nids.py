import os
import sys
import time
import json
import argparse
import traceback
import warnings
import psutil
import numpy as np
import signal
import atexit
from flask import Flask, request, jsonify
import joblib

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="NIDS Inference Service")
parser.add_argument("--model", type=str, default="rf",
                    choices=["rf", "xgb", "lgbm", "svm", "mlp"],
                    help="Модель для загрузки (по умолчанию: rf)")
parser.add_argument("--port", type=int, default=5000,
                    help="Порт Flask (по умолчанию: 5000)")
parser.add_argument("--host", type=str, default="127.0.0.1",
                    help="Хост (по умолчанию: 127.0.0.1; для Docker — '0.0.0.0')")
args = parser.parse_args()

MODEL_NAME = args.model
PORT = args.port
HOST = args.host

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
METRICS_FILE = os.path.join(BASE_DIR, "../data/metrics.json")

prefix = f"{MODEL_NAME}_nids"
METADATA_PATH = os.path.join(MODELS_DIR, f"{prefix}_metadata.json")
PREPROCESSORS_PATH = os.path.join(MODELS_DIR, f"{prefix}_preprocessors.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, f"{prefix}.pkl")

for path, name in [(METADATA_PATH, "метаданные"), (PREPROCESSORS_PATH, "препроцессоры"), (MODEL_PATH, "модель")]:
    if not os.path.exists(path):
        print(f"Ошибка: {name} не найдены: {path}")
        print(f"Выполните: python train_nids.py --model {MODEL_NAME}")
        sys.exit(1)

with open(METADATA_PATH, "r", encoding='utf-8') as f:
    metadata = json.load(f)

preprocessors = joblib.load(PREPROCESSORS_PATH)
model = joblib.load(MODEL_PATH)

feature_names = metadata["feature_names"]
numerical_features = metadata["numerical_features"]
categorical_features = metadata["categorical_features"]
classes = metadata.get("classes", [])
is_binary = (len(classes) == 2)

num_imputer = preprocessors['num_imputer']
scaler = preprocessors['scaler']
cat_imputer = preprocessors.get('cat_imputer')
cat_encoder = preprocessors.get('cat_encoder')

print(f"Модель {MODEL_NAME.upper()} загружена. Признаков: {len(feature_names)}")

app = Flask(__name__)

REQUEST_COUNT = 0
TOTAL_INFERENCE_TIME = 0.0
TOTAL_ATTACKS = 0
INFERENCE_TIMES = []

def save_inference_metrics():
    global REQUEST_COUNT, TOTAL_INFERENCE_TIME, TOTAL_ATTACKS, INFERENCE_TIMES
    if REQUEST_COUNT == 0:
        return
    avg_time = TOTAL_INFERENCE_TIME / REQUEST_COUNT
    min_time = min(INFERENCE_TIMES) if INFERENCE_TIMES else 0.0
    max_time = max(INFERENCE_TIMES) if INFERENCE_TIMES else 0.0

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "language": "python",
        "stage": "inference",
        "host": HOST,
        "port": PORT,
        "requests_total": REQUEST_COUNT,
        "attacks_detected": TOTAL_ATTACKS,
        "inference_time_ms": {
            "mean": round(avg_time, 2),
            "min": round(min_time, 2),
            "max": round(max_time, 2),
            "total_ms": round(TOTAL_INFERENCE_TIME, 2)
        }
    }

    all_metrics = {"python": {}}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        except:
            pass

    if MODEL_NAME not in all_metrics["python"]:
        all_metrics["python"][MODEL_NAME] = []
    all_metrics["python"][MODEL_NAME].append(entry)

    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Метрики инференса сохранены в: {METRICS_FILE}")

def graceful_shutdown(signum, frame):
    print("Получен сигнал завершения. Сохранение метрик...")
    save_inference_metrics()
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
atexit.register(save_inference_metrics)

@app.route('/health', methods=['GET'])
def health():
    global REQUEST_COUNT, TOTAL_INFERENCE_TIME, TOTAL_ATTACKS
    avg_time = (TOTAL_INFERENCE_TIME / REQUEST_COUNT) if REQUEST_COUNT > 0 else 0
    mem_mb = psutil.Process().memory_info().rss / 1024**2
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "requests_total": REQUEST_COUNT,
        "attacks_detected": TOTAL_ATTACKS,
        "avg_inference_time_ms": round(avg_time, 2),
        "memory_mb": round(mem_mb, 1),
        "classes": classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    global REQUEST_COUNT, TOTAL_INFERENCE_TIME, TOTAL_ATTACKS, INFERENCE_TIMES

    try:
        start_time = time.perf_counter()

        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Требуется {'features': {...}}"}), 400
        input_features = data['features']
        if not isinstance(input_features, dict):
            return jsonify({"error": "'features' должен быть объектом"}), 400

        num_values = []
        for feature in numerical_features:
            value = input_features.get(feature, 0.0)
            try:
                if isinstance(value, str):
                    value = float(value.replace(',', '.'))
                elif isinstance(value, bool):
                    value = 1.0 if value else 0.0
                num_values.append(float(value))
            except (ValueError, TypeError):
                num_values.append(0.0)

        cat_encoded = np.empty((1, 0))
        if categorical_features:
            cat_values = [str(input_features.get(f, "Missing")) for f in categorical_features]
            cat_array = np.array(cat_values).reshape(1, -1)
            cat_imputed = cat_imputer.transform(cat_array)
            cat_encoded = cat_encoder.transform(cat_imputed)

        num_array = np.array(num_values).reshape(1, -1)
        num_imputed = num_imputer.transform(num_array)
        num_scaled = scaler.transform(num_imputed)

        features_processed = np.hstack([num_scaled, cat_encoded]) if categorical_features else num_scaled

        if features_processed.shape[1] != len(feature_names):
            return jsonify({"error": f"Несоответствие признаков: {features_processed.shape[1]} vs {len(feature_names)}"}), 400

        pred = model.predict(features_processed)[0]
        proba = None
        try:
            probabilities = model.predict_proba(features_processed)[0]
            proba = probabilities.tolist()
            attack_proba = float(probabilities[1]) if is_binary else float(max(probabilities))
        except:
            attack_proba = 1.0 if pred != classes[0] else 0.0

        inference_time_ms = (time.perf_counter() - start_time) * 1000
        REQUEST_COUNT += 1
        TOTAL_INFERENCE_TIME += inference_time_ms
        INFERENCE_TIMES.append(inference_time_ms)

        is_attack = bool(attack_proba > 0.5 if is_binary else pred != classes[0])
        if is_attack:
            TOTAL_ATTACKS += 1

        pred_label = classes[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)
        print(f"Запрос #{REQUEST_COUNT} | {inference_time_ms:.2f} мс | {pred_label} | p={attack_proba:.4f}")

        response = {
            "prediction_label": pred_label,
            "is_attack": is_attack,
            "inference_time_ms": round(inference_time_ms, 2),
            "model": MODEL_NAME
        }

        if proba is not None:
            if is_binary:
                response.update({
                    "probability_normal": round(proba[0], 6),
                    "probability_attack": round(proba[1], 6)
                })
            else:
                response["class_probabilities"] = {str(cls): round(p, 6) for cls, p in zip(classes, proba)}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({"error": "Требуется {'samples': [{features}, ...]}"}), 400
        results = []
        for sample in data['samples']:
            fake_request = type('obj', (object,), {
                'get_json': lambda: {"features": sample.get("features", {})}
            })
            with app.test_request_context():
                res = predict()
                results.append(res.json if res.status_code == 200 else {"error": res.json})
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/meta', methods=['GET'])
def meta():
    return jsonify(metadata)

if __name__ == '__main__':
    print(f"Сервис запущен: http://{HOST}:{PORT}")
    print("Доступные эндпоинты: /health, /predict (POST), /predict/batch (POST), /meta")
    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
