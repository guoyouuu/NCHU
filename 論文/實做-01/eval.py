"""
eval_full.py — 股票模型完整評估（對齊最新 SeqStockTransformer）
-------------------------------------------------------
功能包含：
1. Daily RankIC（mean / std / IR）
2. IC 時序圖 + IC 分布圖
3. Prediction Distribution
4. Top-K Long/Short 回測
5. Equity Curve
6. Factor Buckets（單調性）
7. Sharpe / 年化報酬 / MDD
8. 輸出 CSV + JSON 統計結果
-------------------------------------------------------
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================
# 1. Command line arguments
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

CKPT_PATH = args.ckpt
TOPK = args.topk
SAVE_DIR = os.path.dirname(CKPT_PATH) or "./eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 2. Dataset（全對齊最新版 prepare_datasets）
# =========================================================
from prepare_model_inputs import prepare_datasets
from sequence_transformer_model import (
    SeqStockTransformer,
    encode_calendar_dates,
)

(
    train_ds, train_loader,
    valid_ds, valid_loader,
    test_ds, test_loader,
    symbols, dates
) = prepare_datasets()

assert test_loader is not None, "❌ test_loader 是 None，代表沒有 test 資料"

# =========================================================
# 3. Load checkpoint
# =========================================================
DEVICE = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

ckpt = torch.load(CKPT_PATH, map_location="cpu")

config    = ckpt["config"]
symbol2id = ckpt["symbol2id"]
year2id   = ckpt["year2id"]

# feature_dim from actual dataset
sample_batch = next(iter(test_loader))
feature_dim = sample_batch["features"].shape[-1]
seq_len = sample_batch["features"].shape[1]

# =========================================================
# 4. Rebuild model (完全對齊 train.py)
# =========================================================
model = SeqStockTransformer(
    feature_dim=feature_dim,
    hidden_dim=config["hidden_dim"],
    nhead=config["nhead"],
    num_layers=config["num_layers"],
    num_symbols=config["num_symbols"],
    sym_emb_dim=config["sym_emb_dim"],
    num_years=config["num_years"],
    year_emb_dim=config["year_emb_dim"],
    max_len=config["seq_len"],
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()


# =========================================================
# 5. Helper encoders（與 train.py 完全相同）
# =========================================================
def encode_symbols(symbol_batch):
    return torch.tensor([symbol2id[s] for s in symbol_batch], dtype=torch.long)

def encode_year_ids(dates_batch):
    out = []
    for seq in dates_batch:
        out.append([year2id[int(d[:4])] for d in seq])
    return torch.tensor(out, dtype=torch.long)

def encode_calendar_seq(dates_batch):
    flat = [d for seq in dates_batch for d in seq]
    cal = encode_calendar_dates(flat)
    B = len(dates_batch)
    L = len(dates_batch[0])
    return cal.view(B, L, -1)


# =========================================================
# 6. Collect predictions
# =========================================================
def collect_predictions(loader):
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Collect Predictions]", ncols=100):

            x = batch["features"].to(DEVICE)
            y = batch["label"].to(DEVICE).squeeze(-1)

            sid  = encode_symbols(batch["symbol"]).to(DEVICE)
            yids = encode_year_ids(batch["dates"]).to(DEVICE)
            cal  = encode_calendar_seq(batch["dates"]).to(DEVICE)

            pred = model(x, sid, yids, cal).cpu().numpy()
            y_np = y.cpu().numpy()

            for s, d, p, l in zip(batch["symbol"], batch["date"], pred, y_np):
                rows.append([s, d, p, l])

    df = pd.DataFrame(rows, columns=["symbol", "date", "pred", "label"])
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna()

df = collect_predictions(test_loader)

# =========================================================
# 7. Daily RankIC
# =========================================================
def compute_daily_ic(df):
    rec = []
    for dt, g in df.groupby("date"):
        if len(g) < 2:
            continue
        pr = g["pred"].rank()
        lb = g["label"].rank()
        if pr.std() == 0 or lb.std() == 0:
            continue
        ic = pr.corr(lb)
        if pd.notna(ic):
            rec.append([dt, ic])
    return pd.DataFrame(rec, columns=["date", "IC"])

ic_df = compute_daily_ic(df)
ic_mean = ic_df["IC"].mean()
ic_std = ic_df["IC"].std()
ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan


# =========================================================
# 8. Top-K portfolio
# =========================================================
def compute_topk(df, k):
    rows = []
    for dt, g in df.groupby("date"):
        if len(g) < k * 2:
            continue
        g = g.sort_values("pred", ascending=False)
        long = g.head(k)["label"].mean()
        short = g.tail(k)["label"].mean()
        rows.append([dt, long, short, long - short])
    return pd.DataFrame(rows, columns=["date", "long", "short", "ls"])

port_df = compute_topk(df, TOPK).sort_values("date")
port_df["cum_ls"] = (1 + port_df["ls"]).cumprod()

# Risk stats
ann = (1 + port_df["ls"].mean()) ** 252 - 1
vol = port_df["ls"].std() * np.sqrt(252)
sharpe = ann / vol if vol > 0 else np.nan
mdd = (port_df["cum_ls"] / port_df["cum_ls"].cummax() - 1).min()


# =========================================================
# 9. Factor buckets（單調性分析）
# =========================================================
def factor_buckets(df, bins=5):
    df2 = df.copy()
    df2["bin"] = pd.qcut(df2["pred"], q=bins, labels=False, duplicates="drop")
    rec = []
    for b, g in df2.groupby("bin"):
        rec.append([b, len(g), g["label"].mean()])
    return pd.DataFrame(rec, columns=["bin", "count", "avg_return"])

fbin_df = factor_buckets(df)


# =========================================================
# 10. Plot functions
# =========================================================
def plot_ic_timeseries(ic_df):
    plt.figure(figsize=(10, 4))
    plt.plot(ic_df["date"], ic_df["IC"])
    plt.title("Daily RankIC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/ic_timeseries.png")
    plt.close()

def plot_ic_distribution(ic_df):
    plt.figure(figsize=(6, 4))
    plt.hist(ic_df["IC"], bins=30)
    plt.title("RankIC Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/ic_hist.png")
    plt.close()

def plot_pred_distribution(df):
    plt.figure(figsize=(6, 4))
    plt.hist(df["pred"], bins=50)
    plt.title("Prediction Value Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/pred_dist.png")
    plt.close()

def plot_equity_curve(port_df):
    plt.figure(figsize=(10, 4))
    plt.plot(port_df["date"], port_df["cum_ls"])
    plt.title("Long-Short Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/equity_curve.png")
    plt.close()

def plot_factor_buckets(fbin_df):
    plt.figure(figsize=(6, 4))
    plt.bar(fbin_df["bin"], fbin_df["avg_return"])
    plt.title("Factor Buckets")
    plt.xlabel("Bin")
    plt.ylabel("Avg Return")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/factor_bins.png")
    plt.close()

# Save all images
plot_ic_timeseries(ic_df)
plot_ic_distribution(ic_df)
plot_pred_distribution(df)
plot_equity_curve(port_df)
plot_factor_buckets(fbin_df)

# =========================================================
# 11. Export CSV + JSON summary
# =========================================================
df.to_csv(f"{SAVE_DIR}/daily_pred.csv", index=False)
ic_df.to_csv(f"{SAVE_DIR}/daily_ic.csv", index=False)
port_df.to_csv(f"{SAVE_DIR}/daily_portfolio.csv", index=False)
fbin_df.to_csv(f"{SAVE_DIR}/factor_bins.csv", index=False)

summary = {
    "ic_mean": float(ic_mean),
    "ic_std": float(ic_std),
    "ic_ir": float(ic_ir),
    "annual_return": float(ann),
    "sharpe": float(sharpe),
    "mdd": float(mdd),
}

json.dump(summary, open(f"{SAVE_DIR}/eval_summary.json", "w"), indent=2)

print("\n🎯 評估完成！所有結果已輸出到：", SAVE_DIR)
print("📈 已生成圖：IC 時序、IC 分布、預測值分布、Equity、分桶")
