"""
원본 데이터 → Δt 포함 train/test 생성

LightGCN 저자 배포본(train.txt/test.txt)에는 타임스탬프가 없으므로 원본에서 직접 재구축한다.

핵심 규칙
--------
1. 유저 단위 서브샘플링 (상호작용 랜덤 샘플링 금지 — 히스토리에 구멍이 나면 시간 순서가 파괴됨)
2. split 은 반드시 시간 순 (유저별 leave-last-N). 랜덤 split 은 미래 정보 누출.
3. Δt 기준점은 train 의 최대 timestamp. test timestamp 는 어떤 경로로도 참조하지 않는다.

사용법
-----
python preprocess.py --dataset gowalla \
    --raw loc-gowalla_totalCheckins.txt --out data/gowalla --user_frac 0.25
"""
import argparse, os, json
import numpy as np
import pandas as pd

SEC_PER_DAY = 86400.0


def load_gowalla(path):
    # SNAP loc-gowalla_totalCheckins.txt : user \t time(ISO) \t lat \t lon \t locid
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["user", "time", "lat", "lon", "item"])
    df["ts"] = pd.to_datetime(df["time"], errors="coerce").astype("int64") // 10**9
    return df[["user", "item", "ts"]].dropna()


def load_amazon(path):
    # Amazon reviews csv : item,user,rating,timestamp  (컬럼 순서는 파일마다 확인 필요)
    df = pd.read_csv(path, header=None, names=["item", "user", "rating", "ts"])
    return df[["user", "item", "ts"]]


def k_core(df, k=10):
    while True:
        uc = df.user.value_counts(); ic = df.item.value_counts()
        keep = df.user.map(uc).ge(k) & df.item.map(ic).ge(k)
        if keep.all():
            return df
        df = df[keep]
        if df.empty:
            raise RuntimeError("k-core 필터링 후 데이터가 비었습니다. k를 낮추세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gowalla", "amazon"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--user_frac", type=float, default=0.25, help="유저 단위 서브샘플링 비율")
    ap.add_argument("--kcore", type=int, default=10)
    ap.add_argument("--test_ratio", type=float, default=0.2, help="유저별 최근 N%를 test로")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_gowalla(args.raw) if args.dataset == "gowalla" else load_amazon(args.raw)
    print(f"[raw] {len(df):,} interactions, {df.user.nunique():,} users, {df.item.nunique():,} items")

    # (u,i) 중복 상호작용 → 가장 최근 것만 유지 (Δt 정의를 하나로 확정)
    df = df.sort_values("ts").drop_duplicates(subset=["user", "item"], keep="last")

    # 유저 단위 서브샘플링 → 선택된 유저의 히스토리는 전부 보존
    if args.user_frac < 1.0:
        users = df.user.unique()
        pick = rng.choice(users, size=int(len(users) * args.user_frac), replace=False)
        df = df[df.user.isin(pick)]
        print(f"[subsample] user_frac={args.user_frac} → {df.user.nunique():,} users")

    df = k_core(df, args.kcore)
    print(f"[{args.kcore}-core] {len(df):,} interactions, "
          f"{df.user.nunique():,} users, {df.item.nunique():,} items")

    umap = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    imap = {v: i for i, v in enumerate(sorted(df.item.unique()))}
    df["u"] = df.user.map(umap); df["i"] = df.item.map(imap)

    # 시간 순 split (유저별 최근 test_ratio 를 test 로)
    df = df.sort_values(["u", "ts"])
    rank = df.groupby("u").cumcount()
    size = df.groupby("u")["u"].transform("size")
    is_test = rank >= (size * (1 - args.test_ratio)).astype(int)
    tr, te = df[~is_test], df[is_test]

    # Δt : train 최대 timestamp 기준 경과일. test 는 절대 참조하지 않음.
    t_ref = int(tr.ts.max())
    dt_days = ((t_ref - tr.ts) / SEC_PER_DAY).clip(lower=0)

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame({"u": tr.u.values, "i": tr.i.values,
                  "dt_days": dt_days.values}).to_csv(f"{args.out}/train.csv", index=False)
    pd.DataFrame({"u": te.u.values, "i": te.i.values}).to_csv(f"{args.out}/test.csv", index=False)

    meta = {"dataset": args.dataset, "n_users": len(umap), "n_items": len(imap),
            "n_train": len(tr), "n_test": len(te), "t_ref": t_ref,
            "dt_max_days": float(dt_days.max()), "dt_median_days": float(dt_days.median()),
            "user_frac": args.user_frac, "kcore": args.kcore, "seed": args.seed,
            "density": len(tr) / (len(umap) * len(imap))}
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))

    # 코사인 유사도 측정용 고정 노드쌍 — 모든 실행에서 동일해야 곡선 비교가 성립
    N = len(umap) + len(imap)
    pairs = rng.integers(0, N, size=(5000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    np.save(f"{args.out}/cos_pairs.npy", pairs)
    print(f"[cos_pairs] {len(pairs)} 쌍 고정 저장 → 전 모델·전 K 에서 이 파일을 공유할 것")


if __name__ == "__main__":
    main()
