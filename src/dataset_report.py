"""
데이터셋 리포트 — 팀 공유 · 발표 슬라이드용

meta.json 은 기계용이라 사람이 보기 불편하고, Δt 분포가 빠져 있다.
이 스크립트가 발표에 그대로 넣을 수 있는 표와 그림을 만든다.

python dataset_report.py --data data/gowalla --out figs
"""
import argparse, json, hashlib, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# 한글 폰트가 있으면 사용, 없으면 그림 라벨은 영문으로 (Colab 기본 환경엔 한글 폰트 없음).
# Colab 에서 한글 그림을 원하면:  !apt-get install -y fonts-nanum && rm -rf ~/.cache/matplotlib
_installed = {f.name for f in font_manager.fontManager.ttflist}
KO = next((f for f in ["NanumGothic", "Malgun Gothic", "AppleGothic",
                       "Noto Sans CJK KR", "NanumBarunGothic"] if f in _installed), None)
if KO:
    plt.rcParams["font.family"] = KO
    plt.rcParams["axes.unicode_minus"] = False
L = (lambda ko, en: ko) if KO else (lambda ko, en: en)   # 라벨 선택기

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="figs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

meta = json.load(open(f"{a.data}/meta.json"))
tr = pd.read_csv(f"{a.data}/train.csv")
te = pd.read_csv(f"{a.data}/test.csv")
dt = tr.dt_days.values
upc = tr.groupby("u").size()
ipc = tr.groupby("i").size()

# ── 데이터 지문 : 팀원 간 동일 데이터 사용 확인용 ──────────────────────
h = hashlib.md5()
for f in ["train.csv", "test.csv", "cos_pairs.npy"]:
    h.update(open(f"{a.data}/{f}", "rb").read())
fingerprint = h.hexdigest()[:12]

lines = []
P = lines.append
P("=" * 62)
P(f"  데이터셋 리포트 — {meta['dataset']}")
P(f"  데이터 지문(fingerprint): {fingerprint}")
P("  ※ 팀원 전원의 지문이 같아야 결과를 합칠 수 있습니다.")
P("=" * 62)
P("")
P("[규모]")
P(f"  유저 수              {meta['n_users']:>12,}")
P(f"  아이템 수            {meta['n_items']:>12,}")
P(f"  train 상호작용       {meta['n_train']:>12,}")
P(f"  test 상호작용        {meta['n_test']:>12,}")
P(f"  density              {meta['density']:>12.6f}   ({meta['density']*100:.4f}%)")
P(f"  {meta['kcore']}-core, 유저 단위 서브샘플링 {meta['user_frac']*100:.0f}%, seed {meta['seed']}")
P("")
P("[유저별 상호작용 수 — 희소성 분포]")
P(f"  min / p25 / median / p75 / p95 / max")
P(f"  {upc.min()} / {upc.quantile(.25):.0f} / {upc.median():.0f} / "
  f"{upc.quantile(.75):.0f} / {upc.quantile(.95):.0f} / {upc.max()}")
P(f"  아이템별: median {ipc.median():.0f}, p95 {ipc.quantile(.95):.0f}, max {ipc.max()}")
P("")
P("[Δt 분포 — 이 프로젝트의 전제]")
P(f"  기준점: train 최대 timestamp (test 는 미참조)")
P(f"  min      {dt.min():>10.1f} 일")
P(f"  p25      {np.percentile(dt,25):>10.1f} 일")
P(f"  median   {np.median(dt):>10.1f} 일")
P(f"  p75      {np.percentile(dt,75):>10.1f} 일")
P(f"  p95      {np.percentile(dt,95):>10.1f} 일")
P(f"  max      {dt.max():>10.1f} 일   ({dt.max()/365:.1f} 년)")
P("")
P("[λ 후보별 α 스케일 — 감쇠가 실제로 작동할 범위 확인]")
P(f"  {'λ':>8} {'반감기':>10} {'α(p25)':>9} {'α(중앙)':>9} {'α(p95)':>9} {'퍼짐':>8}")
for lam in [0.0005, 0.001, 0.005, 0.01, 0.05]:
    a25, a50, a95 = [np.exp(-lam*x) for x in
                     (np.percentile(dt,25), np.median(dt), np.percentile(dt,95))]
    P(f"  {lam:>8} {np.log(2)/lam:>9.0f}일 {a25:>9.3f} {a50:>9.3f} {a95:>9.3f} {a25-a95:>8.3f}")
P("  → '퍼짐'이 0.05 미만인 구간에서는 α가 사실상 상수라 시간 정보가 작동하지 않습니다.")
P("")

report = "\n".join(lines)
print(report)
open(f"{a.out}/dataset_report.txt", "w").write(report)

# ── 그림 : Δt 분포 + 유저 활동량 분포 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))

axes[0].hist(dt, bins=60, color="steelblue", edgecolor="white", lw=0.3)
axes[0].axvline(np.median(dt), color="crimson", ls="--", lw=1.2,
                label=f"median {np.median(dt):.0f}")
axes[0].set(xlabel=L("Δt (일)", "Δt (days)"), ylabel=L("엣지 수", "# edges"),
            title=L("Δt 분포 — 시간 신호가 존재하는가", "Δt distribution — is there a time signal?"))
axes[0].legend(fontsize=7)

xs = np.linspace(0, dt.max(), 300)
for lam in [0.001, 0.005, 0.01, 0.05]:
    axes[1].plot(xs, np.exp(-lam*xs), lw=1.4, label=f"λ={lam}")
axes[1].set(xlabel=L("Δt (일)", "Δt (days)"), ylabel="α = exp(−λΔt)", ylim=(0, 1.05),
            title=L("λ 후보별 감쇠 곡선", "Decay curves by λ"))
axes[1].legend(fontsize=7)

axes[2].hist(upc.values, bins=50, color="seagreen", edgecolor="white", lw=0.3)
axes[2].set(xlabel=L("유저별 train 상호작용 수", "interactions per user (train)"),
            ylabel=L("유저 수", "# users"), yscale="log",
            title=L("유저 활동량 분포", "User activity distribution"))

fig.tight_layout()
fig.savefig(f"{a.out}/fig0_dataset.png")
print(f"저장: {a.out}/dataset_report.txt, {a.out}/fig0_dataset.png")
