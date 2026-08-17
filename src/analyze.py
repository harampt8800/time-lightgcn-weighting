"""
결과 회수 후 분석 — 발표용 그림 3개 + 가설 판정표 생성

python analyze.py --runs runs --out figs
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# Colab 기본 환경엔 한글 폰트가 없어 라벨이 깨짐 → 있으면 쓰고 없으면 영문.
# 한글 그림이 필요하면: !apt-get install -y fonts-nanum && rm -rf ~/.cache/matplotlib
_inst = {f.name for f in font_manager.fontManager.ttflist}
KO = next((f for f in ["NanumGothic","Malgun Gothic","AppleGothic",
                       "Noto Sans CJK KR","NanumBarunGothic"] if f in _inst), None)
if KO:
    plt.rcParams["font.family"] = KO
    plt.rcParams["axes.unicode_minus"] = False
L = (lambda ko, en: ko) if KO else (lambda ko, en: en)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="figs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

files = glob.glob(f"{a.runs}/*.csv")
if not files:
    raise SystemExit(f"{a.runs} 에 결과 csv 가 없습니다.")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
ev = df.dropna(subset=["recall@20"])                      # 평가가 돌아간 epoch만
best = ev.loc[ev.groupby("run_id")["recall@20"].idxmax()]  # run별 best epoch

# ── 그림 1 : λ 학습 곡선 (H2 의 본체) ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.6))
for rid, g in df[df["mode"].isin(["mul", "twadd"])].groupby("run_id"):
    ax.plot(g.epoch, g["lambda"], lw=1.4, label=rid)
ax.axhline(0, color="crimson", ls="--", lw=1)
ax.text(0.01, 0.03, L("λ ≤ 0 으로 수렴 → H2 반증", "λ ≤ 0 → H2 refuted"),
        transform=ax.transAxes, color="crimson", fontsize=8)
ax.set(xlabel="epoch", ylabel="λ", title="Learned time-decay λ  (H2)")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(f"{a.out}/fig1_lambda_curve.png"); plt.close(fig)

# ── 그림 2 : α 분포 ──────────────────────────────────────────────
sub = best[best["mode"].isin(["mul", "twadd"])]
if len(sub) and "alpha_p50" in sub:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    y = np.arange(len(sub))
    ax.hlines(y, sub["alpha_p05"], sub["alpha_p95"], lw=6, alpha=0.35, color="teal")
    ax.plot(sub["alpha_p50"], y, "o", color="teal", label="median")
    ax.plot(sub["alpha_mean"], y, "x", color="darkorange", label="mean")
    ax.set_yticks(y); ax.set_yticklabels(sub["run_id"], fontsize=6)
    ax.set(xlabel="α = exp(−λ·Δt)",
           title=L("학습된 α 분포 (p05–p95) — 폭이 좁으면 사실상 무효과",
                   "Learned α spread (p05–p95) — narrow = no real effect"))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig2_alpha_dist.png"); plt.close(fig)

# ── 그림 3 : K 곡선 + 코사인 유사도 (H3) ────────────────────────────
cos_cols = sorted([c for c in best.columns if c.startswith("cos_sim_")],
                  key=lambda c: int(c.split("_")[-1]))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for mode, g in best.groupby("mode"):
    g = g.sort_values("K")
    axes[0].plot(g.K, g["recall@20"], "o-", label=mode)
    deep = g[g.K == g.K.max()]
    if len(deep) and cos_cols:
        axes[1].plot(range(len(cos_cols)), deep[cos_cols].iloc[0].values, "o-", label=mode)
axes[0].set(xlabel="K (layers)", ylabel="Recall@20",
            title=L("K 곡선 — 정점이 이동했는가 (H3)", "Recall vs K — did the peak shift? (H3)"))
axes[1].set(xlabel="layer", ylabel="mean cosine similarity",
            title=L("오버스무딩 — 고정 5000쌍", "Over-smoothing — fixed 5000 pairs"))
for ax in axes: ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{a.out}/fig3_K_and_oversmoothing.png"); plt.close(fig)

# ── 가설 판정표 ─────────────────────────────────────────────────
cols = ["run_id", "mode", "dataset", "K", "seed", "recall@20", "ndcg@20",
        "lambda", "half_life_days"]
tbl = best[[c for c in cols if c in best.columns]].sort_values(["dataset", "mode", "K"])
tbl.to_csv(f"{a.out}/results_table.csv", index=False)
print(tbl.to_string(index=False))

print("\n=== 가설 판정 보조 ===")
lam = best[best["mode"].isin(["mul", "twadd"])]["lambda"]
if len(lam):
    print(f"H2 : 학습된 λ 범위 {lam.min():+.5f} ~ {lam.max():+.5f}  "
          f"→ {'양수 수렴, H2 지지' if lam.min() > 0 else 'λ≤0 존재, H2 반증 검토'}")
for ds, g in best.groupby("dataset"):
    for mode, gg in g.groupby("mode"):
        if gg.K.nunique() >= 3:
            peak = gg.loc[gg["recall@20"].idxmax(), "K"]
            print(f"H3 : [{ds}] {mode:8s} 정점 K = {peak}")
print("\n※ mul 의 정점이 뒤로 밀렸다면, constc / alpha_renorm 결과와 반드시 대조할 것.")
print("   대조군도 함께 밀렸다면 원인은 시간 정보가 아니라 임베딩 크기 축소입니다.")
