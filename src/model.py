"""
Time-LightGCN — 시간 정보 주입 LightGCN (가중치 파트)

모드
----
vanilla : 기준점. e_u^(k+1) = Σ 1/√(|N_u||N_i|) · e_i^(k)
add     : 1-hop만.  e_u^(1) = Σ norm · ( e_i^(0) + β·Φ(Δt) )        β = 학습 스칼라
mul     : 매 layer. e_u^(k+1) = Σ norm · α(Δt) · e_i^(k)            α = exp(-λ·Δt), λ 학습
twadd   : 1-hop만.  e_u^(1) = Σ norm · ( e_i^(0) + α(Δt)·Φ(Δt) )
constc  : 매 layer. e_u^(k+1) = Σ norm · c · e_i^(k)                c = 상수 (H3 대조군)

핵심 설계 결정
-------------
1. λ 초기값 = 0  →  α = exp(0) = 1  →  mul/twadd가 정확히 vanilla와 동일한 상태에서 출발.
   "우리는 바닐라에서 시작했고, 데이터가 λ를 양수로 밀어냈다"는 방어 논리의 근거.
2. λ에 부호 제약 없음 (softplus 미사용). λ ≤ 0 으로 수렴하면 H2가 스스로 반증됨.
3. 정규화 항 1/√(|N_u||N_i|) 는 원본 인접행렬 기준으로 고정.
   α는 그 위에 곱해질 뿐 degree 계산에 개입하지 않음 → 모드 간 비교가 α 효과로만 격리됨.
4. Φ는 L2 정규화하여 ‖Φ‖=1 로 맞춤. (sin/cos raw 는 d차원에서 ‖Φ‖=√(d/2) 라
   발표자료의 '‖Φ‖=1' 전제와 어긋남 — 여기서 정규화로 해결.)
"""

import math
import numpy as np
import torch
import torch.nn as nn

EXP_CLAMP = 8.0  # exp 인자 클램프. λ가 음수로 갈 때 발산 방지.


def build_time_encoding_table(max_days: int, dim: int) -> torch.Tensor:
    """Δt(정수 일) → Φ 룩업 테이블. shape (max_days+1, dim), 각 행 L2 norm = 1.

    엣지마다 Φ를 저장하면 메모리가 터지므로 '일' 단위로 버킷팅해 테이블로 관리한다.
    """
    pos = torch.arange(max_days + 1, dtype=torch.float32).unsqueeze(1)  # (T,1)
    idx = torch.arange(0, dim, 2, dtype=torch.float32)                  # (dim/2,)
    div = torch.exp(-math.log(10000.0) * idx / dim)                     # (dim/2,)
    table = torch.zeros(max_days + 1, dim)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    table = table / table.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return table


class TimeLightGCN(nn.Module):
    def __init__(self, n_users, n_items, edges_u, edges_i, edges_dt_days,
                 dim=64, K=3, mode="vanilla", const_c=1.0, alpha_renorm=False,
                 device="cpu"):
        """
        edges_u, edges_i : train 상호작용의 유저/아이템 인덱스 (길이 E, 각각 0-base)
        edges_dt_days    : 해당 상호작용의 Δt (일 단위, float 또는 int, >= 0)
        alpha_renorm     : True 면 α를 목적지 노드별 평균 1이 되도록 재정규화.
                           α<1 을 매 layer 곱하면 깊은 layer 크기가 지수적으로 붕괴하는데,
                           그러면 "K가 커져도 성능 유지"라는 H3 결과가
                           '시간 정보 덕분'인지 '그냥 깊은 layer를 죽인 것'인지 구별 불가.
                           재정규화하면 크기는 보존되고 '비균일성'만 남아
                           H3 의 메커니즘 주장을 정확히 검증할 수 있다.
        """
        super().__init__()
        assert mode in ("vanilla", "add", "mul", "twadd", "constc")
        self.n_users, self.n_items = n_users, n_items
        self.N = n_users + n_items
        self.dim, self.K, self.mode = dim, K, mode
        self.const_c = float(const_c)
        self.alpha_renorm = bool(alpha_renorm)
        self.device = device

        self.emb = nn.Embedding(self.N, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

        # --- 학습 파라미터 -------------------------------------------------
        # λ: 초기 0 → α=1 → vanilla와 동일 출발. 부호 제약 없음.
        self.lam = nn.Parameter(torch.tensor(0.0))
        # β: add 전용 스케일 스칼라. Φ(‖Φ‖=1)를 임베딩 스케일(≈0.1)로 낮추는 역할.
        self.beta = nn.Parameter(torch.tensor(0.1))

        self._build_graph(edges_u, edges_i, edges_dt_days)

        if mode in ("add", "twadd"):
            max_days = int(self.edge_dt_bucket.max().item())
            self.register_buffer("phi_table", build_time_encoding_table(max_days, dim))
        else:
            self.phi_table = None

    # ------------------------------------------------------------------
    def _build_graph(self, eu, ei, edt):
        eu = torch.as_tensor(np.asarray(eu), dtype=torch.long)
        ei = torch.as_tensor(np.asarray(ei), dtype=torch.long) + self.n_users
        edt = torch.as_tensor(np.asarray(edt), dtype=torch.float32).clamp(min=0.0)

        # 양방향 (u->i, i->u). 같은 상호작용이므로 Δt는 두 방향에서 동일.
        src = torch.cat([eu, ei])
        dst = torch.cat([ei, eu])
        dt = torch.cat([edt, edt])

        deg = torch.zeros(self.N).index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg = deg.clamp(min=1.0)
        norm = (deg[src].pow(-0.5) * deg[dst].pow(-0.5))

        self.register_buffer("edge_src", src)
        self.register_buffer("edge_dst", dst)
        self.register_buffer("edge_norm", norm)
        self.register_buffer("edge_dt", dt)
        self.register_buffer("edge_dt_bucket", dt.round().long())

    # ------------------------------------------------------------------
    def alpha(self) -> torch.Tensor:
        """엣지별 시간 감쇠 계수 α(Δt) = exp(-λ·Δt)."""
        a = torch.exp(torch.clamp(-self.lam * self.edge_dt, min=-EXP_CLAMP, max=EXP_CLAMP))
        if self.alpha_renorm:
            # 목적지 노드별 평균 α = 1 로 맞춤 → 총 유입 크기 보존, 비균일성만 유지
            s = torch.zeros(self.N, device=a.device).index_add_(0, self.edge_dst, a)
            cnt = torch.zeros(self.N, device=a.device).index_add_(
                0, self.edge_dst, torch.ones_like(a))
            mean = (s / cnt.clamp(min=1.0)).clamp(min=1e-8)
            a = a / mean[self.edge_dst]
        return a

    def half_life_days(self):
        """발표용 해석 지표: α가 0.5가 되는 경과일. λ<=0 이면 정의되지 않음."""
        lam = self.lam.item()
        return math.log(2.0) / lam if lam > 1e-8 else float("inf")

    # ------------------------------------------------------------------
    def propagate(self, return_layers=False):
        x = self.emb.weight
        src, dst, norm = self.edge_src, self.edge_dst, self.edge_norm

        if self.mode == "mul":
            w = self.alpha()
        elif self.mode == "constc":
            w = torch.full_like(norm, self.const_c)
        else:
            w = None

        outs = [x]
        for k in range(self.K):
            msg = x[src]

            # 1-hop 에만 시간 벡터 주입 (2-hop 부터의 '남의 시간' 오염 차단)
            if k == 0 and self.mode in ("add", "twadd"):
                phi = self.phi_table[self.edge_dt_bucket]          # (2E, d)
                s = self.beta if self.mode == "add" else self.alpha().unsqueeze(-1)
                msg = msg + s * phi

            if w is not None:
                msg = msg * w.unsqueeze(-1)

            msg = msg * norm.unsqueeze(-1)
            x = torch.zeros(self.N, self.dim, device=msg.device, dtype=msg.dtype)
            x = x.index_add_(0, dst, msg)
            outs.append(x)

        final = torch.stack(outs, dim=0).mean(dim=0)   # α_k = 1/(K+1) 균등, LightGCN 원본과 동일
        if return_layers:
            return final, outs
        return final

    def user_item_embeddings(self):
        e = self.propagate()
        return e[: self.n_users], e[self.n_users:]

    # ------------------------------------------------------------------
    def bpr_loss(self, users, pos, neg, reg=1e-4):
        eu, ei = self.user_item_embeddings()
        u, p, n = eu[users], ei[pos], ei[neg]
        scores = (u * p).sum(-1) - (u * n).sum(-1)
        loss = -torch.nn.functional.logsigmoid(scores).mean()

        e0 = self.emb.weight
        reg_term = (e0[users].pow(2).sum()
                    + e0[pos + self.n_users].pow(2).sum()
                    + e0[neg + self.n_users].pow(2).sum()) / (2 * len(users))
        return loss + reg * reg_term, loss.item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def layer_norms(self):
        """레이어별 임베딩 평균 L2 크기. H3 교란요인(크기 붕괴) 진단용 — 반드시 로깅할 것."""
        _, outs = self.propagate(return_layers=True)
        return [h.norm(dim=1).mean().item() for h in outs]

    @torch.no_grad()
    def alpha_stats(self):
        """학습된 α의 분포. 전부 ~1 이거나 전부 ~0 이면 사실상 아무 일도 안 일어난 것."""
        if self.mode not in ("mul", "twadd"):
            return {}
        a = self.alpha()
        q = torch.quantile(a, torch.tensor([0.05, 0.5, 0.95], device=a.device))
        return {"alpha_min": a.min().item(), "alpha_p05": q[0].item(),
                "alpha_p50": q[1].item(), "alpha_p95": q[2].item(),
                "alpha_max": a.max().item(), "alpha_mean": a.mean().item()}

    @torch.no_grad()
    def layer_cosine_similarity(self, pair_idx):
        """레이어별 임베딩 코사인 유사도 (오버스무딩 측정, H3).

        pair_idx : (P, 2) long tensor — 전 실행에서 동일한 고정 쌍이어야 비교가 성립.
        반환      : 길이 K+1 리스트, layer 0..K 각각의 평균 코사인 유사도
        """
        _, outs = self.propagate(return_layers=True)
        res = []
        for h in outs:
            a = torch.nn.functional.normalize(h[pair_idx[:, 0]], dim=1)
            b = torch.nn.functional.normalize(h[pair_idx[:, 1]], dim=1)
            res.append((a * b).sum(1).mean().item())
        return res
