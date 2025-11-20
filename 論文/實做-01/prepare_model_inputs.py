# =====================================================
# 🔧 可調參數 / 路徑設定（全部集中在最上方）
# =====================================================

MERGED_FEATURES_PATH = "./data/qlib_data/day1/generated_datasets/merged_features.parquet"

LOOKBACK = 60               # 過去 N 天
BATCH_SIZE = 64
NUM_WORKERS = 4


META_COLS = ["symbol", "date", "year", "phase"]
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
NON_FEATURE_COLS = ["VWAP0", "label"]

EXCLUDE_COLS = set(META_COLS + OHLCV_COLS + NON_FEATURE_COLS)

# =====================================================
#  🚀 下面開始為 Dataset / Loader 主邏輯
# =====================================================

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# # =====================================================
# # 🔍 根據 merged_features 自動推算 train/valid/test 年度界線
# # =====================================================
# def infer_date_boundaries(df: pd.DataFrame):
#     """
#     從 df 的 phase=test 自動推算:
#     - train_end
#     - valid_end
#     - test_end
#     - train_year / valid_year / test_year（如需更進階用途）
#     """

#     test_year = df[df["phase"] == "test"]["date"].max().year
#     valid_year = df[df["phase"] == "valid"]["date"].max().year
#     train_year = df[df["phase"] == "train"]["date"].max().year  
    
#     print(f"Train year: {train_year}, Valid year: {valid_year}, Test year: {test_year}")   
#     return (
#         f"{train_year}-12-31",
#         f"{valid_year}-12-31",
#         f"{test_year}-12-31",
#         train_year,
#         valid_year,
#         test_year
#     )

# =====================================================
# 🧩 自訂 collate_fn：避免 dates 被 PyTorch 亂 transpose
# =====================================================
def seq_collate_fn(batch):
    return {
        "symbol":   [b["symbol"] for b in batch],          # list[str], B
        "dates":    [b["dates"]  for b in batch],          # list[list[str]], B×L
        "date":     [b["date"]   for b in batch],          # list[str], B  ← ★ 新增
        "features": torch.stack([b["features"] for b in batch]),  # [B, L, F]
        "label":    torch.stack([b["label"]   for b in batch]),   # [B, 1]
    }

# =====================================================
# 🧱 LazySeqDataset：跨年度 window + 標準化 sample 建立邏輯
# =====================================================
class LazySeqDataset(Dataset):
    """
    - subset 用日期切（避免看到未來資料）
    - sample 所屬階段由 label 的 phase 決定（避免洩漏）
    - window 自動由 lookback 控制，不會抓過遠資料
    """

    def __init__(self, df, feature_cols, lookback, phase):
        self.df = df.sort_values(["symbol", "date"])
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.phase = phase
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        """滑動 window 建立 sample"""
        for symbol, g in self.df.groupby("symbol"):
            g = g.reset_index(drop=True)

            for end in range(self.lookback - 1, len(g)):
                start = end - self.lookback + 1

                # label 所屬 phase → 決定 sample 屬於 train / valid / test
                if g.iloc[end]["phase"] != self.phase:
                    continue

                self.samples.append((symbol, g, start, end))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        symbol, g, start, end = self.samples[idx]

        window = g.iloc[start: end + 1]
        assert len(window) == self.lookback

        features = window[self.feature_cols].to_numpy(dtype="float32")
        label = window.iloc[-1]["label"]
        label_date = window.iloc[-1]["date"]

        dates = window["date"].astype(str).tolist()

        return {
            "symbol": symbol,                
            "dates": dates,                               
            "features": torch.tensor(features),
            "date": str(label_date),
            "label": torch.tensor([label], dtype=torch.float32),
        }

# =====================================================
# 🧩 建立 train / valid / test Dataset + DataLoader
# =====================================================
def build_phase_dataset(df, phase_name, feature_cols,
                        lookback, batch_size, num_workers):
    ds = LazySeqDataset(df, feature_cols, lookback, phase_name)

    if len(ds) == 0:
        print(f"⚠ Phase '{phase_name}' 無資料")
        return None, None
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(phase_name == "train"),
        num_workers=num_workers,
        drop_last=False,
        collate_fn=seq_collate_fn,
    )

    return ds, loader


# =====================================================
# 🎯 主入口：供 train.py / eval.py 呼叫
# =====================================================
def prepare_datasets(
    merged_path=MERGED_FEATURES_PATH,
    lookback=LOOKBACK,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
):
    print("準備資料中…")

    df = pd.read_parquet(merged_path)
    df = df.sort_values(["symbol", "date"])
    print(df["symbol"].unique().shape[0], "支股票")
    
    # 🔥 過濾資料太少的股票
    df = df.groupby("symbol").filter(lambda x: len(x) >= lookback)
    symbols = sorted(df["symbol"].unique().tolist())
    dates = sorted(df["date"].unique().tolist())
    print("過濾後剩下", len(symbols), "支股票")

    # ----------- 🔥 自動建立 label（必要） -----------
    df["label"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1

    # ----------- 🔥 處理缺值（必要） -----------
    df = df.fillna(0)

    # ---- 自動抓 feature columns ----
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
    ]
    print(f"Detected {len(feature_cols)} feature columns.")
    print(f"Feature columns: {feature_cols}")
    
    train_ds, train_loader = build_phase_dataset(
        df, "train", feature_cols, lookback, batch_size, num_workers
    )

    valid_ds, valid_loader = build_phase_dataset(
        df, "valid", feature_cols, lookback, batch_size, num_workers
    )

    test_ds, test_loader = build_phase_dataset(
        df, "test", feature_cols, lookback, batch_size, num_workers
    )

    print("Train samples:", len(train_ds))
    print("Valid samples:", len(valid_ds))
    print("Test samples :", len(test_ds))

    return (
        train_ds, train_loader,
        valid_ds, valid_loader,
        test_ds, test_loader,
        symbols, dates
    )


# =====================================================
# 🧪 工具：檢查 Dataset 中任意一筆 sample
# =====================================================
def get_last_sample_of_symbol(ds, symbol):
    """
    回傳 Dataset 中某支股票最後一筆 sample 的 index
    並回傳該 sample 內容
    """
    # 找出所有 sample 的 index
    matched = []
    for i in range(len(ds)):
        if ds.samples[i][0] == symbol:
            matched.append(i)

    if len(matched) == 0:
        print(f"❌ Symbol {symbol} not found in this dataset.")
        return None

    # 最後一筆 sample 的 index = 最大的 sample index
    last_idx = matched[-1]

    print(f"✔ Symbol {symbol} 最後一筆 sample index = {last_idx}")
    return last_idx, ds[last_idx]

def inspect_sample(ds, index=0, n_show_rows=20):
    if ds is None or len(ds) == 0:
        print("❌ Dataset is empty.")
        return

    sample = ds[index]
    print("Sample: ", sample)

    symbol = sample["symbol"]
    t_date = sample["date"]
    window_dates = sample["dates"]
    features = sample["features"].numpy()
    label = float(sample["label"].numpy())

    # 找到完整序列
    g = ds.df[ds.df["symbol"] == symbol].sort_values("date").reset_index(drop=True)

    # 找到 label row index
    idx = g.index[g["date"] == t_date][0]

    start_idx = idx - (ds.lookback - 1)
    end_idx = idx
    window_df = g.iloc[start_idx:end_idx + 1]

    print("\n================= SAMPLE DETAIL (LOOKBACK =", ds.lookback, ") =================\n")

    print(f"▶ Sample index     : {index}")
    print(f"▶ Symbol           : {symbol}")
    print(f"▶ Label date (t)   : {t_date}")
    print(f"▶ Window dates     : {window_dates}")
    print(f"▶ Label(t→t+1 rtn) : {label:.6f}")
    print(f"▶ Label phase      : {g.loc[idx, 'phase']}")

    print("\n▶ Window size      :", len(window_df))
    print("\n--- Window DataFrame with date, OHLCV and label ---")
    print(window_df[META_COLS + OHLCV_COLS + ["label"]])    
    print(f"▶ Window date range: {window_df['date'].iloc[0]} → {window_df['date'].iloc[-1]}")

    # 洩漏檢查
    leakage = (window_df["date"] > t_date).any()
    print(f"▶ 未來資料洩漏？   : {'❌YES' if leakage else '✔NO'}")

    print("\n--- Time Line ---")
    print(f"[window] {window_df['date'].iloc[0]} → ... → {window_df['date'].iloc[-1]} → [label] {t_date} → t+1")

    print("\n--- Feature matrix shape ---")
    print(f"{features.shape}   (應為 {ds.lookback} x {len(ds.feature_cols)})")

    print("\n====================================================================\n")

# =====================================================
# 🧪 工具：檢查 dataset 中所有 window 是否完整
# =====================================================
def check_all_windows(ds):
    """
    全集 window 健康檢查：
    - 確認 window size 是否完整
    - 找出所有可能洩漏未來資料的 sample
    - 找出跨 phase window（合法，但提供告警）
    """

    print("\n==================== WINDOW CONSISTENCY CHECK ====================")

    problems = []
    leakage_cnt = 0
    cross_phase_cnt = 0

    total = len(ds)
    L = ds.lookback

    for i in range(total):
        symbol, g, start, end = ds.samples[i]

        # label row
        t_date = g.iloc[end]["date"]
        t_phase = g.iloc[end]["phase"]

        # window rows
        w = g.iloc[start:end+1]
        window_size = len(w)

        # 1. 檢查 window 長度
        if window_size != L:
            problems.append((i, symbol, t_date, window_size))
            continue

        # 2. 未來洩漏檢查
        if (w["date"] > t_date).any():
            leakage_cnt += 1

        # 3. 跨 phase 檢查（合法，但顯示統計）
        #   window 舊資料 phase 可以是 train，label phase 是 valid/test → 合法
        #   如果 window 中有比 label phase 更「未來」的資料 → 違法（第 2 步已擋）
        phases_in_window = set(w["phase"].unique())
        if t_phase not in phases_in_window:
            cross_phase_cnt += 1

    print(f"✔ Dataset: {total} samples")
    print(f"✔ Lookback: {L}")
    
    if problems:
        print("\n❌ Window 缺資料樣本:")
        for (idx, sym, date, ws) in problems[:20]:
            print(f"  - Sample {idx} | {sym} @ {date} | window rows = {ws} (should be {L})")
        print(f"... 共 {len(problems)} 筆不完整 window")
    else:
        print("✔ 所有 window 都有完整長度")

    if leakage_cnt > 0:
        print(f"\n❌ 未來資料洩漏: 共 {leakage_cnt} 筆（嚴重）")
    else:
        print("✔ 無未來資料洩漏")

    print(f"\n⚠ 跨 phase window（正常現象，用於時間序列）: {cross_phase_cnt} 筆")
    print("   （例如 valid sample 需要 1991 的 train 資料 → 合法）")

    print("\n==================================================================\n")


# =====================================================
# ✔ 測試執行（你可移除）
# =====================================================
if __name__ == "__main__":
    train_ds, train_loader, valid_ds, valid_loader, test_ds, test_loader, symbols, dates = prepare_datasets()


    # 看 train sample
    idx, _ = get_last_sample_of_symbol(train_ds, "1101")
    inspect_sample(train_ds, 0)
    inspect_sample(train_ds, idx)


    # 看 valid sample
    idx, _ = get_last_sample_of_symbol(valid_ds, "1101")
    inspect_sample(valid_ds, 0)
    inspect_sample(valid_ds, idx)

    # 看 test sample
    idx, _ = get_last_sample_of_symbol(test_ds, "1101")
    inspect_sample(test_ds, 0)
    inspect_sample(test_ds, idx)

    check_all_windows(train_ds)
    check_all_windows(valid_ds)
    check_all_windows(test_ds)