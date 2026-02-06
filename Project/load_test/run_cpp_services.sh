#!/bin/bash
set -e

MODELS=("rf" "svm" "mlp")
PORT_BASE=6000
PIDS=()

echo "Запуск C++-сервисов..."
for i in "${!MODELS[@]}"; do
    model="${MODELS[i]}"
    port=$((PORT_BASE + i))
    ../cpp/nids serve --model "$model" --port "$port" > "/tmp/nids_cpp_$model.log" 2>&1 &
    PIDS+=($!)
    echo "  $model → порт $port (PID ${PIDS[-1]})"
    sleep 0.5
done

echo "Ожидание готовности (15 сек)..."
sleep 15

echo "Запуск нагрузочного теста..."
python3 test_inference_cpp.py

echo "Остановка сервисов..."
kill ${PIDS[@]} 2>/dev/null || true
wait ${PIDS[@]} 2>/dev/null || true

echo "Готово"
