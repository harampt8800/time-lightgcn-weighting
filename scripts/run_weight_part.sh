#!/usr/bin/env bash
# 가중치 파트 실험 전체. Colab 세션이 끊겨도 재실행하면 체크포인트에서 자동 재개됩니다.
set -e
DATA=${DATA:-data/gowalla}
SEED=${SEED:-0}

echo "=== 사전 검증 ==="
python src/sanity_check.py --data "$DATA"

echo "=== H3 곡선: mul K=1..5 ==="
for K in 1 2 3 4 5; do
  python src/train.py --data "$DATA" --mode mul --K $K --seed $SEED
done

echo "=== H2: t-w add K=3,5 ==="
for K in 3 5; do
  python src/train.py --data "$DATA" --mode twadd --K $K --seed $SEED
done

echo "=== H3 대조군 (crucial) ==="
C=$(python -c "import json;print(json.load(open('runs/$(basename $DATA)_mul_K5_s${SEED}_final.json'))['alpha_mean'])")
echo "학습된 alpha_mean = $C 를 상수 c 로 사용"
python src/train.py --data "$DATA" --mode constc --const_c "$C" --K 5 --seed $SEED
python src/train.py --data "$DATA" --mode mul --alpha_renorm --K 5 --seed $SEED

echo "=== 분석 ==="
python src/analyze.py --runs runs --out figs
