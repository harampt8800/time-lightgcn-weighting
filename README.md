# Time-LightGCN — 시간 가중 α(Δt) 모듈

LightGCN의 선형 전파 구조를 유지한 채 **엣지별 시간 감쇠 가중치**를 주입하고,
그 효과를 반증 가능한 형태로 검증하는 구현체입니다.

> **이 레포의 담당 범위**
> 팀 프로젝트 "Time-LightGCN" 중 **가중치(α) 파트** — 시간 감쇠 함수 설계, mul/t-w add
> 모델 구현, H2·H3 검증 파이프라인, 오버스무딩 교란요인 대조군.
> 단순합(add, β 스칼라) 파트와 발표 자료는 다른 팀원 담당입니다.

---

## 핵심 기여

**1. 반증 가능하게 설계된 시간 감쇠 함수**

```
α(Δt) = exp(−λ · Δt),   λ ∈ ℝ (학습),  λ_init = 0,  부호 제약 없음
```

λ=0이면 α=1이므로 **바닐라 LightGCN과 수치적으로 완전히 동일한 지점에서 출발**합니다.
`sanity_check.py`가 이를 max|diff| = 0으로 증명합니다. 부호를 강제하지 않으므로
λ ≤ 0으로 수렴하면 가설이 스스로 반증됩니다 — "최근성"을 함수에 심어놓지 않았습니다.

**2. 오버스무딩 주장의 교란요인 발견과 대조군 설계**

α<1을 매 layer 곱하면 깊은 layer의 임베딩 크기가 지수적으로 붕괴합니다
(λ=0.01, K=4에서 바닐라 대비 **1/150**). 그러면 "K가 커져도 성능이 유지된다"는 결과가
시간 정보 덕분인지 **단지 깊은 layer를 죽인 것인지 구별할 수 없습니다.**

이를 분리하기 위해 대조군 두 종을 구현했습니다.

| 대조군 | 무엇을 통제하는가 |
|---|---|
| `--mode constc --const_c <ᾱ>` | 시간과 무관한 상수 곱. 이것도 버티면 원인은 크기 축소 → **가설 기각** |
| `--mode mul --alpha_renorm` | α를 노드별 평균 1로 재정규화. 크기 보존, **비균일성만** 잔존 → 여기서도 완화되면 가설 강화 |

**3. 재현성 장치**

데이터 지문(fingerprint) 기반 팀원 간 데이터셋 일치 검증, 고정 노드쌍 코사인 유사도,
체크포인트 자동 재개(Colab 12시간 세션 대응), epoch 단위 λ 로깅.

---

## 검증 가설

| | 내용 | 판정 근거 |
|---|---|---|
| **H2** | 최근 상호작용에 더 큰 가중치를 주면 단순 시점 구별보다 낫다 | 학습된 λ의 부호 + t-w add vs add |
| **H3** | 시간 가중이 집계를 비균일하게 만들어 오버스무딩을 완화한다 | K 곡선 정점 이동 + 레이어별 코사인 유사도 (**대조군 필수**) |

---

## 구현 모델

| 모드 | 수식 | 주입 위치 |
|---|---|---|
| `vanilla` | `Σ 1/√(\|N_u\|\|N_i\|) · e_i` | — |
| `add` | `Σ norm · (e_i + β·Φ(Δt))` | 1-hop |
| `mul` | `Σ norm · α(Δt) · e_i` | 매 layer |
| `twadd` | `Σ norm · (e_i + α(Δt)·Φ(Δt))` | 1-hop |
| `constc` | `Σ norm · c · e_i` | 매 layer (H3 대조군) |

`add` / `twadd`가 동일하게 1-hop에만 주입되므로, 둘의 차이는 **오직 "β 상수 vs α 함수"**로
격리됩니다. H2 비교가 공정해지는 지점입니다.

---

## 실행

```bash
pip install -r requirements.txt

# 1) 전처리 — 한 사람만 실행하고 data/ 폴더를 통째로 공유할 것
python src/preprocess.py --dataset gowalla \
       --raw loc-gowalla_totalCheckins.txt --out data/gowalla --user_frac 0.25

# 2) 데이터셋 리포트 — 팀원 전원의 fingerprint가 같아야 결과를 합칠 수 있음
python src/dataset_report.py --data data/gowalla --out figs

# 3) 사전 검증 — 통과 못 하면 본실험 금지
python src/sanity_check.py --data data/gowalla

# 4) 본실험
bash scripts/run_weight_part.sh

# 5) 분석
python src/analyze.py --runs runs --out figs
```

---

## 산출물

| 파일 | 내용 |
|---|---|
| `figs/fig0_dataset.png` | Δt 분포 · λ별 감쇠 곡선 · 유저 활동량 |
| `figs/fig1_lambda_curve.png` | **λ 학습 곡선 — H2의 본체** |
| `figs/fig2_alpha_dist.png` | 학습된 α 분포 (폭이 좁으면 사실상 무효과) |
| `figs/fig3_K_and_oversmoothing.png` | **K 곡선 + 오버스무딩 — H3의 본체 (대조군 포함)** |
| `figs/results_table.csv` | 가설 판정표 |

---

## 문서

- [`docs/design_rationale.md`](docs/design_rationale.md) — 가중치 설계 근거 (발표 방어용)
- [`docs/logging_spec.md`](docs/logging_spec.md) — 팀 인계용 로깅 명세
- [`docs/h3_confound.md`](docs/h3_confound.md) — 크기 붕괴 교란요인 분석

---

## 참고

- He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR 2020
- Yang et al., *GraphPro: Graph Pre-training and Prompt Learning for Recommendation*, WWW 2024
- Zhou et al., *Layer-refined Graph Convolutional Networks for Recommendation*, 2022
- Kim et al., *Revisiting LightGCN: Inflexibility, Inconsistency, and A Remedy*, RecSys 2024
