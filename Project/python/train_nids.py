import pandas as pd
import numpy as np
import time
import os
import json
import warnings
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Train NIDS model on NF-UNSW-NB15")
parser.add_argument("--model", type=str, default="rf",
                    choices=["rf", "xgb", "lgbm", "svm", "mlp"],
                    help="Модель для обучения: rf, xgb, lgbm, svm, mlp (default: rf)")
parser.add_argument("--sample", type=int, default=50_000,
                    help="Макс. размер выборки перед обучением (default: 50000)")
parser.add_argument("--data", type=str, default="../data/NF-UNSW-NB15-v2.csv",
                    help="Путь к CSV-файлу (default: ../data/NF-UNSW-NB15-v2.csv)")
args = parser.parse_args()

MODEL_NAME = args.model
SAMPLE_SIZE = args.sample
DATA_PATH = args.data

print("="*60)
print(f"TRAIN NIDS | Модель: {MODEL_NAME.upper()} | Выборка: {SAMPLE_SIZE:,}")
print("="*60)

print(f"Загружаем данные из {DATA_PATH}...")
start_load = time.time()
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл не найден: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
load_time = time.time() - start_load
print(f"Загружено {df.shape[0]:,} строк, {df.shape[1]} колонок за {load_time:.2f} сек")

target_col = None
candidates = ['Label', 'label', 'target', 'class', 'attack']
for cand in candidates:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    raise ValueError(f"Не найдена целевая колонка. Возможные имена: {candidates}. Фактические: {list(df.columns)}")

print(f"Целевая переменная: '{target_col}'")
class_counts = df[target_col].value_counts()
print(f"Распределение классов:\n{class_counts}")

print("Отбор признаков...")
cols_to_drop = []
for col in df.columns:
    if col == target_col:
        continue
    if col.lower() in ['timestamp', 'date', 'time', 'src_ip', 'dst_ip', 'attack_id', 'attack_cat', 'attack_name']:
        cols_to_drop.append(col)
        continue
    if df[col].nunique() > 0.5 * len(df):
        cols_to_drop.append(col)
        continue
    if df[col].isnull().mean() > 0.3:
        cols_to_drop.append(col)

feature_cols = [col for col in df.columns if col not in cols_to_drop + [target_col]]
print(f"Удалено {len(cols_to_drop)} колонок: {cols_to_drop[:5]}{'...' if len(cols_to_drop)>5 else ''}")
print(f"Осталось {len(feature_cols)} признаков")

numerical_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
categorical_cols = [col for col in feature_cols if col not in numerical_cols]

print(f"Числовых: {len(numerical_cols)} | Категориальных: {len(categorical_cols)}")

if numerical_cols:
    print(f"  Примеры числовых: {numerical_cols[:3]}")
if categorical_cols:
    print(f"  Примеры категориальных: {categorical_cols[:3]}")
print(f"Выборка ({SAMPLE_SIZE:,} макс.) и балансировка...")
if len(df) > SAMPLE_SIZE:
    minority = class_counts.idxmin()
    majority = class_counts.idxmax()
    minority_count = class_counts.min()
    
    if minority_count < 1000 and len(class_counts) == 2:
        print("Обнаружен сильный дисбаланс → undersampling majority класса (max 20x minority)")
        df_minority = df[df[target_col] == minority]
        df_majority = df[df[target_col] == majority]
        n_majority = min(len(df_majority), minority_count * 20)
        df_majority_sampled = df_majority.sample(n=n_majority, random_state=42)
        df = pd.concat([df_minority, df_majority_sampled], ignore_index=True)
        print(f"→ После undersampling: {len(df)} строк")
    else:
        print("→ Стратифицированная случайная выборка")
        _, df = train_test_split(df, test_size=SAMPLE_SIZE/len(df), stratify=df[target_col], random_state=42)

print(f"Итоговый размер данных: {len(df)}")
print(f"Распределение после:\n{df[target_col].value_counts()}")

y = df[target_col]
X = df[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

print("Предобработка данных...")

num_imputer = SimpleImputer(strategy='median')
X_train_num = num_imputer.fit_transform(X_train[numerical_cols])
X_test_num = num_imputer.transform(X_test[numerical_cols])

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

if categorical_cols:
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test_cat = cat_imputer.transform(X_test[categorical_cols])
    
    X_train_cat_encoded = cat_encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = cat_encoder.transform(X_test_cat)
    
    X_train_processed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num_scaled, X_test_cat_encoded])
    
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        for cat in cat_encoder.categories_[i]:
            cat_feature_names.append(f"{col}_{cat}")
    feature_names = numerical_cols + cat_feature_names
else:
    X_train_processed = X_train_num_scaled
    X_test_processed = X_test_num_scaled
    feature_names = numerical_cols

print(f"После препроцессинга: {X_train_processed.shape[1]} признаков")

def get_model(model_name):
    n_classes = len(np.unique(y_train))
    is_binary = (n_classes == 2)
    
    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    elif model_name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic' if is_binary else 'multi:softprob',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary' if is_binary else 'multiclass',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_name == "svm":
        from sklearn.svm import SVC
        return SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

print(f"Обучение модели: {MODEL_NAME.upper()}")
model = get_model(MODEL_NAME)

start_time = time.time()
model.fit(X_train_processed, y_train)
train_time = time.time() - start_time
print(f"Обучение завершено за {train_time:.2f} сек")

print("Оценка на тестовой выборке...")
y_pred = model.predict(X_test_processed)

acc = accuracy_score(y_test, y_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

roc_auc = None
try:
    proba = model.predict_proba(X_test_processed)
    if len(np.unique(y_train)) == 2:
        roc_auc = roc_auc_score(y_test, proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, proba, multi_class='ovr')
except Exception as e:
    pass

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {acc:.4f}")
print(f"Macro  — Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f} | F1: {f1_macro:.4f}")
print(f"Weighted — Precision: {precision_weighted:.4f} | Recall: {recall_weighted:.4f} | F1: {f1_weighted:.4f}")
if roc_auc is not None:
    print(f"ROC-AUC: {roc_auc:.4f}")

print("По классам:")
print(classification_report(y_test, y_pred, digits=4))

print("Сохранение модели и артефактов...")
os.makedirs("../models", exist_ok=True)
os.makedirs("../load_test", exist_ok=True)
os.makedirs("../data", exist_ok=True)

prefix = f"../models/{MODEL_NAME}_nids"

joblib.dump(model, f"{prefix}.pkl")
print(f"Модель: {prefix}.pkl")

preprocessors = {
    'num_imputer': num_imputer,
    'scaler': scaler,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'feature_names': feature_names
}
if categorical_cols:
    preprocessors.update({
        'cat_imputer': cat_imputer,
        'cat_encoder': cat_encoder
    })
joblib.dump(preprocessors, f"{prefix}_preprocessors.pkl")
print(f"Препроцессоры: {prefix}_preprocessors.pkl")

metadata = {
    "model_type": MODEL_NAME,
    "input_shape": [1, X_train_processed.shape[1]],
    "feature_names": feature_names,
    "numerical_features": numerical_cols,
    "categorical_features": categorical_cols,
    "target": target_col,
    "classes": sorted(y_train.unique().tolist()),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "metrics": {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(roc_auc) if roc_auc is not None else None
    },
    "training_time_sec": train_time
}
with open(f"{prefix}_metadata.json", "w", encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Метаданные: {prefix}_metadata.json")

preprocessing_params = {
    "numerical_features": numerical_cols,
    "categorical_features": categorical_cols,
    "numerical_medians": {},
    "numerical_means": {},
    "numerical_scales": {},
    "categorical_categories": {}
}

for i, col in enumerate(numerical_cols):
    if hasattr(num_imputer, 'statistics_') and i < len(num_imputer.statistics_):
        preprocessing_params["numerical_medians"][col] = float(num_imputer.statistics_[i])
    if hasattr(scaler, 'mean_') and i < len(scaler.mean_):
        preprocessing_params["numerical_means"][col] = float(scaler.mean_[i])
    if hasattr(scaler, 'scale_') and i < len(scaler.scale_):
        preprocessing_params["numerical_scales"][col] = float(scaler.scale_[i])

if categorical_cols and hasattr(cat_encoder, 'categories_'):
    for i, col in enumerate(categorical_cols):
        cats = cat_encoder.categories_[i].tolist()
        preprocessing_params["categorical_categories"][col] = [str(c) for c in cats]

with open(f"{prefix}_preprocessing_params.json", "w", encoding='utf-8') as f:
    json.dump(preprocessing_params, f, indent=2, ensure_ascii=False)
print(f"Параметры препроцессинга: {prefix}_preprocessing_params.json")

sample_data = {}
for col in numerical_cols:
    if col in X_test.columns:
        sample_data[col] = float(X_test[col].iloc[0])
for col in categorical_cols:
    if col in X_test.columns:
        sample_data[col] = str(X_test[col].iloc[0])

payload = {"features": sample_data}
with open("../load_test/payload_nids.json", "w", encoding='utf-8') as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(f"Пример запроса: ../load_test/payload_nids.json")

METRICS_FILE = "../data/metrics.json"
entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": MODEL_NAME,
    "language": "python",
    "data_path": DATA_PATH,
    "sample_size_used": len(df),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "training_time_sec": train_time,
    "metrics": {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(roc_auc) if roc_auc is not None else None
    }
}

all_metrics = {"python": {}}
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

if MODEL_NAME not in all_metrics["python"]:
    all_metrics["python"][MODEL_NAME] = []
all_metrics["python"][MODEL_NAME].append(entry)

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print(f"Метрики добавлены в: {METRICS_FILE}")

print("Экспорт в ONNX...")
try:
    if MODEL_NAME == "rf":
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, X_train_processed.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=15,
                                     options={id(model): {'zipmap': False}})
        with open(f"{prefix}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX (skl2onnx): {prefix}.onnx")
    elif MODEL_NAME == "xgb":
        from onnxmltools import convert_xgboost
        from onnxconverter_common.data_types import FloatTensorType as ONNXFloatTensorType
        initial_type = [('input', ONNXFloatTensorType([None, X_train_processed.shape[1]]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=15)
        with open(f"{prefix}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX (xgboost): {prefix}.onnx")
    elif MODEL_NAME == "mlp":
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, X_train_processed.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=15,
                                     options={id(model): {'zipmap': False}})
        with open(f"{prefix}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX (MLP): {prefix}.onnx")
    else:
        print(f"ONNX не поддерживается для {MODEL_NAME}. Используйте .pkl.")
except Exception as e:
    print(f"Ошибка ONNX-экспорта: {e}")

print("="*60)
print("ГОТОВО!")
print(f"Модель: {MODEL_NAME.upper()} | Accuracy: {acc:.4f} | Время: {train_time:.1f} сек")
print("="*60)
print(f"Для запуска сервиса: python app_nids.py --model {MODEL_NAME}")
print("="*60)
