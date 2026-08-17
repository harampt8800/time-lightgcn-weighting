"""
학습 · 평가 · 로깅

로깅 명세 (팀 합의 사항 — 이걸 안 남기면 H2/H3 판정 불가, 재실행해야 함)
--------------------------------------------------------------------
epoch_log.csv : run_id, mode, dataset, K, seed, epoch,
                lambda, half_life_days, beta,
                train_loss, recall@20, ndcg@20,
                alpha_min/p05/p50/p95/max/mean,
                layer_norm_0..K, cos_sim_0..K
final.json    : 위의 best-epoch 스냅샷 + 학습된 λ·β 최종값

사용법
-----
python train.py --data data/gowalla --mode mul --K 3 --seed 0
python train.py --data data/gowalla --mode constc --const_c 0.42 --K 3 --seed 0   # H3 대조군
python train.py --data data/gowalla --mode mul --alpha_renorm --K 3 --seed 0      # 크기 교란 제거판
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from model import TimeLightGCN


# ---------------------------------------------------------------- 평가
@torch.no_grad()
def evaluate(model, train_pos_csr, test_dict, topk=20, batch=1024):
    eu, ei = model.user_item_embeddings()
    users = sorted(test_dict.keys())
    recalls, ndcgs = [], []
    ideal = np.cumsum(1.0 / np.log2(np.arange(2, topk + 2)))

    for s in range(0, len(users), batch):
        chunk = users[s:s + batch]
        scores = eu[chunk] @ ei.T
        for r, u in enumerate(chunk):                    # train 아이템 마스킹
            seen = train_pos_csr.get(u)
            if seen is not None and len(seen):
                scores[r, seen] = -1e9
        top = torch.topk(scores, topk, dim=1).indices.cpu().numpy()

        for r, u in enumerate(chunk):
            gt = test_dict[u]
            hit = np.isin(top[r], list(gt)).astype(np.float64)
            recalls.append(hit.sum() / len(gt))
            dcg = (hit / np.log2(np.arange(2, topk + 2))).sum()
            ndcgs.append(dcg / ideal[min(len(gt), topk) - 1])
    return float(np.mean(recalls)), float(np.mean(ndcgs))


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["vanilla", "add", "mul", "twadd", "constc"], required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5, help="eval 단위 early stopping")
    ap.add_argument("--const_c", type=float, default=1.0)
    ap.add_argument("--alpha_renorm", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(f"{args.data}/meta.json"))
    nu, ni = meta["n_users"], meta["n_items"]
    tr = pd.read_csv(f"{args.data}/train.csv")
    te = pd.read_csv(f"{args.data}/test.csv")
    pairs = torch.as_tensor(np.load(f"{args.data}/cos_pairs.npy"), dtype=torch.long, device=dev)

    train_pos = {u: g.i.values for u, g in tr.groupby("u")}
    test_dict = {u: set(g.i.values) for u, g in te.groupby("u")}

    tag = f"{meta['dataset']}_{args.mode}_K{args.K}_s{args.seed}"
    if args.alpha_renorm: tag += "_renorm"
    if args.mode == "constc": tag += f"_c{args.const_c}"
    os.makedirs(args.out, exist_ok=True)
    log_path, ckpt_path = f"{args.out}/{tag}.csv", f"{args.out}/{tag}.pt"

    model = TimeLightGCN(nu, ni, tr.u.values, tr.i.values, tr.dt_days.values,
                         dim=args.dim, K=args.K, mode=args.mode,
                         const_c=args.const_c, alpha_renorm=args.alpha_renorm).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, best, bad, rows = 0, -1.0, 0, []
    if os.path.exists(ckpt_path):                        # Colab 세션 끊김 대비 재개
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        rows = ck["rows"]
        print(f"[resume] epoch {start_epoch} 부터 재개")

    u_all = torch.as_tensor(tr.u.values, dtype=torch.long, device=dev)
    i_all = torch.as_tensor(tr.i.values, dtype=torch.long, device=dev)
    E = len(u_all)
    t0 = time.time()

    for ep in range(start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(E, device=dev)
        tot = 0.0
        for s in range(0, E, args.batch):
            idx = perm[s:s + args.batch]
            neg = torch.randint(0, ni, (len(idx),), device=dev)   # uniform negative 1개 (LightGCN 원본)
            loss, raw = model.bpr_loss(u_all[idx], i_all[idx], neg, reg=args.reg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += raw * len(idx)
        train_loss = tot / E

        # ---- λ 는 매 epoch 기록 (H2 판정 근거. 최종값만으론 수렴 여부를 못 보임) ----
        row = {"run_id": tag, "mode": args.mode, "dataset": meta["dataset"],
               "K": args.K, "seed": args.seed, "epoch": ep,
               "lambda": model.lam.item(), "half_life_days": model.half_life_days(),
               "beta": model.beta.item(), "train_loss": train_loss,
               "recall@20": None, "ndcg@20": None}

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            rec, ndcg = evaluate(model, train_pos, test_dict)
            row["recall@20"], row["ndcg@20"] = rec, ndcg
            row.update(model.alpha_stats())
            for k, v in enumerate(model.layer_norms()):
                row[f"layer_norm_{k}"] = v
            for k, v in enumerate(model.layer_cosine_similarity(pairs)):
                row[f"cos_sim_{k}"] = v
            print(f"[{tag}] ep{ep:3d} loss={train_loss:.4f} R@20={rec:.4f} "
                  f"N@20={ndcg:.4f} λ={model.lam.item():+.5f} "
                  f"({time.time()-t0:.0f}s)")
            if rec > best:
                best, bad = rec, 0
                json.dump({**row, "best": True}, open(f"{args.out}/{tag}_final.json", "w"), indent=2)
            else:
                bad += 1

        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": best, "bad": bad, "rows": rows}, ckpt_path)

        if bad >= args.patience:
            print(f"[{tag}] early stop @ epoch {ep} (best R@20={best:.4f})")
            break

    print(f"[{tag}] done. best Recall@20 = {best:.4f}, 최종 λ = {model.lam.item():+.5f}, "
          f"반감기 = {model.half_life_days():.1f}일")


if __name__ == "__main__":
    main()
