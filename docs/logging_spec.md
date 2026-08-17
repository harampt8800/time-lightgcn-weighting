# 로깅 명세 — 팀 인계용

**λ를 안 남기면 H2를 판정할 수 없고 전 실험을 재실행해야 합니다.**
`src/train.py` 가 자동으로 남기므로 로깅 부분은 수정하지 마십시오.

## runs/{tag}.csv — 매 epoch 한 행

| 컬럼 | 필요한 이유 |
|---|---|
| `run_id`, `mode`, `dataset`, `K`, `seed`, `epoch` | 식별자 |
| `lambda` | **H2 판정의 핵심.** 최종값만으론 수렴 여부를 못 보임 → 매 epoch |
| `half_life_days` | λ의 발표용 해석 지표. "반감기 69일"이 "λ=0.01"보다 잘 전달됨 |
| `beta` | add 와의 H2 공정 비교용 |
| `train_loss`, `recall@20`, `ndcg@20` | 성능 |
| `alpha_min/p05/p50/p95/max/mean` | α가 실제로 퍼져 있는지. 전부 ~1이면 무효과 |
| `layer_norm_0..K` | **H3 교란요인 진단** (docs/h3_confound.md) |
| `cos_sim_0..K` | 오버스무딩. 고정 5000쌍 기준 |

`recall@20` 등 평가 지표는 `--eval_every` 주기에만 채워지고 나머지 행은 비어 있습니다.
`analyze.py` 가 알아서 걸러냅니다.

## runs/{tag}_final.json

best epoch 스냅샷. `constc` 대조군의 `--const_c` 값을 여기서 꺼냅니다.

## 반드시 공유해야 하는 것

- **`data/<dataset>/` 폴더 전체** — 각자 전처리하면 서브샘플링이 달라져 결과를 합칠 수 없음
- `cos_pairs.npy` — 전 모델·전 K 가 동일한 쌍을 써야 곡선 비교가 성립
- `dataset_report.py` 가 출력하는 **fingerprint** — 팀원 전원 동일해야 함

## 팀원과 맞춰야 할 것

H2 는 add 와 t-w add 의 차이가 **오직 'β 상수 vs α 함수'** 여야만 성립합니다.
아래가 어긋나면 H2 비교표가 무의미해집니다.

- Φ 함수 구현 (`model.build_time_encoding_table` 하나를 공유)
- Δt 정의와 단위 (train 최대 timestamp 기준, 일 단위)
- 데이터 split, seed, early stopping 기준
- 임베딩 차원, 학습률, 정규화 계수
