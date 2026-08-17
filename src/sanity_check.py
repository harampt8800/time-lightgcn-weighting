"""
본실험 전 필수 검증 — 이거 통과 못 하면 20회 돌려도 전부 폐기입니다.

python sanity_check.py --data data/gowalla
"""
import argparse, json
import numpy as np, pandas as pd, torch
from model import TimeLightGCN

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--K", type=int, default=3)
a = ap.parse_args()

meta = json.load(open(f"{a.data}/meta.json"))
tr = pd.read_csv(f"{a.data}/train.csv")
kw = dict(n_users=meta["n_users"], n_items=meta["n_items"],
          edges_u=tr.u.values, edges_i=tr.i.values, edges_dt_days=tr.dt_days.values,
          dim=64, K=a.K)


def make(mode, **extra):
    torch.manual_seed(0)                     # 동일 초기 임베딩 보장
    return TimeLightGCN(mode=mode, **kw, **extra)


ok = True

# 1) λ=0 → mul 이 vanilla 와 정확히 일치해야 함
v, m = make("vanilla"), make("mul")
with torch.no_grad():
    d = (v.propagate() - m.propagate()).abs().max().item()
print(f"[1] λ=0 → mul == vanilla   max|diff| = {d:.3e}  {'OK' if d < 1e-6 else 'FAIL'}")
ok &= d < 1e-6

# 2) β=0 → add 가 vanilla 와 일치
ad = make("add")
with torch.no_grad():
    ad.beta.fill_(0.0)
    d = (v.propagate() - ad.propagate()).abs().max().item()
print(f"[2] β=0 → add == vanilla   max|diff| = {d:.3e}  {'OK' if d < 1e-6 else 'FAIL'}")
ok &= d < 1e-6

# 3) c=1 → constc 가 vanilla 와 일치
cc = make("constc", const_c=1.0)
with torch.no_grad():
    d = (v.propagate() - cc.propagate()).abs().max().item()
print(f"[3] c=1 → constc == vanilla max|diff| = {d:.3e}  {'OK' if d < 1e-6 else 'FAIL'}")
ok &= d < 1e-6

# 4) λ 에 gradient 가 실제로 흐르는가 (detach 실수 검출)
m2 = make("mul")
with torch.no_grad():
    m2.lam.fill_(0.01)
u = torch.randint(0, meta["n_users"], (256,))
p = torch.randint(0, meta["n_items"], (256,))
n = torch.randint(0, meta["n_items"], (256,))
loss, _ = m2.bpr_loss(u, p, n)
loss.backward()
g = m2.lam.grad
print(f"[4] λ gradient = {g.item():+.3e}  {'OK' if g is not None and g.abs() > 0 else 'FAIL'}")
ok &= (g is not None and g.abs().item() > 0)

# 5) α 분포가 실제로 퍼져 있는가 (전부 ~1 이거나 전부 ~0 이면 Δt 스케일이 잘못됨)
st = m2.alpha_stats()
print(f"[5] λ=0.01 일 때 α 분포: p05={st['alpha_p05']:.3f} p50={st['alpha_p50']:.3f} "
      f"p95={st['alpha_p95']:.3f}")
spread = st["alpha_p95"] - st["alpha_p05"]
print(f"    퍼짐 = {spread:.3f}  {'OK' if spread > 0.05 else 'FAIL — Δt 단위(일) 확인 필요'}")
ok &= spread > 0.05

# 6) H3 교란요인 확인 — mul 의 layer 별 크기 붕괴 정도
print("\n[6] layer 별 임베딩 평균 크기 (H3 교란요인 진단)")
mr = make("mul", alpha_renorm=True)
with torch.no_grad():
    mr.lam.fill_(0.01)
print("    layer :  vanilla    mul      mul+renorm")
for k, (x, y, z) in enumerate(zip(v.layer_norms(), m2.layer_norms(), mr.layer_norms())):
    print(f"      {k}   : {x:8.4f} {y:8.4f} {z:8.4f}")
print("    → mul 만 급감하면 'K 유지'가 시간정보 때문인지 크기붕괴 때문인지 구별 불가.")
print("      constc 대조군 또는 --alpha_renorm 결과가 반드시 함께 있어야 H3 주장이 성립.")

print("\n" + ("전체 통과 — 본실험 진행 가능" if ok else "실패 항목 있음 — 본실험 금지"))
