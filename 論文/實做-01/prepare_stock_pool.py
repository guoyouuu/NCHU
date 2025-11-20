import pandas as pd
from pathlib import Path

# === 1️⃣ 檔案路徑設定 ===
SUMMARY_FILE = Path("./data/day1/summary_data.csv")
CALENDAR_FILE = Path("./data/day1/weigt.csv")  # 加權指數交易日曆
OUTPUT_DIR = Path("./data/stock_pools")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# === 2️⃣ 超參數設定 ===
PRETRAIN_WINDOW_YEARS = 5   # 初始預訓練視窗
STEP_YEARS = 1              # 每次滾動年數
MIN_STOCKS_REQUIRED = 50    # 最低股票數門檻

# === 3️⃣ 載入資料 ===
summary_df = pd.read_csv(SUMMARY_FILE, parse_dates=["start_date", "end_date"], low_memory=False)
calendar_df = pd.read_csv(CALENDAR_FILE, parse_dates=["k_datetime"]).sort_values("k_datetime")

# === 4️⃣ 清理資料 ===
summary_df["error"] = summary_df["error"].fillna("")
summary_df["has_non_trading_dates"] = summary_df["has_non_trading_dates"].fillna(0)

valid_stocks = summary_df[
    (summary_df["error"] == "") &
    (summary_df["has_non_trading_dates"] == 0) &
    (summary_df["zero_volume_ratio_file"] < 0.1)
].copy()

print(f"✅ 通過品質篩選的股票數：{len(valid_stocks)}")

# === 5️⃣ 市場整體日期範圍 ===
global_start_date = valid_stocks["start_date"].min()
global_end_date = valid_stocks["end_date"].max()
print(f"📈 市場整體日期範圍：{global_start_date.date()} ~ {global_end_date.date()}")

# === 6️⃣ 初始化視窗設定 ===
current_start_date = global_start_date
current_end_date = (global_start_date + pd.DateOffset(years=PRETRAIN_WINDOW_YEARS)).replace(month=12, day=31)

# # === 7️⃣ 主流程：依年度產生 Incremental Pretrain + 對應 Online Fine-tune ===
# while current_end_date < global_end_date:

#     # === Incremental Pretrain 股票池 ===
#     mask_pretrain = (
#         (valid_stocks["start_date"] <= current_end_date) &
#         (valid_stocks["end_date"] >= current_end_date)
#     )
#     pretrain_subset = valid_stocks.loc[mask_pretrain].copy()

#     if len(pretrain_subset) < MIN_STOCKS_REQUIRED:
#         print(f"⚪ {current_start_date.year}-{current_end_date.year}: 股票不足（{len(pretrain_subset)} 檔），跳過")
#         current_end_date += pd.DateOffset(years=STEP_YEARS)
#         continue

#     # 評估資料完整性（coverage score）
#     pretrain_subset["coverage_score"] = (
#         (1 - pretrain_subset["missing_ratio_ipo"]) *
#         (1 - pretrain_subset["zero_volume_ratio_file"]) * 100
#     )
#     pretrain_subset.sort_values("file", inplace=True)

#     # 輸出 Incremental Pretrain 股票清單
#     pretrain_dir = OUTPUT_DIR / "incremental_pretrain"
#     pretrain_dir.mkdir(exist_ok=True, parents=True)

#     pretrain_file = pretrain_dir / f"{current_end_date.year}.csv"
#     pretrain_subset[["file", "start_date", "end_date", "coverage_score"]].to_csv(pretrain_file, index=False)
#     print(f"🟢 Incremental Pretrain {current_start_date.year}-{current_end_date.year} → {len(pretrain_subset)} 檔")

#     # === Online Fine-tuning 股票池 ===
#     fine_tune_year = current_end_date.year + 1
#     ft_start = pd.Timestamp(f"{fine_tune_year}-01-01")
#     ft_end = min(pd.Timestamp(f"{fine_tune_year}-12-31"), global_end_date)


#     trading_days = calendar_df.query("(@ft_start <= k_datetime <= @ft_end)")["k_datetime"].tolist()
#     if not trading_days:
#         print(f"⚠️ 無 {fine_tune_year} 年交易日，跳過 Online FT")
#         current_end_date += pd.DateOffset(years=STEP_YEARS)
#         continue

#     ft_dir = OUTPUT_DIR / f"online_ft/{fine_tune_year}"
#     ft_dir.mkdir(exist_ok=True, parents=True)

#     daily_summary = []

#     for day in trading_days:
#         mask_day = (valid_stocks["start_date"] <= day) & (valid_stocks["end_date"] >= day)
#         day_stocks = valid_stocks.loc[mask_day, ["file", "start_date", "end_date"]].copy()
#         day_stocks.sort_values("file", inplace=True)
#         new_listed_count = (day_stocks["start_date"] == day).sum()

#         # 儲存每日可交易股票清單
#         day_stocks.to_csv(ft_dir / f"{day.date()}.csv", index=False)

#         daily_summary.append({
#             "date": day.date(),
#             "num_stocks": len(day_stocks),
#             "new_listed": new_listed_count
#         })

#     # 輸出每日 summary 統計表
#     daily_summary_df = pd.DataFrame(daily_summary)
#     summary_file = ft_dir / f"summary_{fine_tune_year}.csv"
#     daily_summary_df.to_csv(summary_file, index=False)

#     print(f"   🧾 Online FT {fine_tune_year}: 交易日 {len(trading_days)} 天, 股票數統計輸出完成 ({summary_file})")

#     # === 更新視窗 ===
#     current_end_date += pd.DateOffset(years=STEP_YEARS)

# print("\n✅ 已完成 Incremental Pretrain + Online Fine-tuning 全流程生成。")

# === 7️⃣ 主流程：依年度產生 Incremental Pretrain（train / valid） + 對應 Online Fine-tune ===
while current_end_date < global_end_date:

    # === Train 股票池（過去 PRETRAIN_WINDOW_YEARS）===
    train_year = current_end_date.year - 1
    train_end = pd.Timestamp(f"{train_year}-12-31")

    mask_train = (
        (valid_stocks["start_date"] <= train_end) &
        (valid_stocks["end_date"] >= current_start_date)
    )
    train_subset = valid_stocks.loc[mask_train].copy()

    if len(train_subset) < MIN_STOCKS_REQUIRED:
        print(f"⚪ {current_start_date.year}-{current_end_date.year}: 股票不足（{len(train_subset)} 檔），跳過")
        current_end_date += pd.DateOffset(years=STEP_YEARS)
        continue

    # === Valid 股票池（該年度資料）===
    valid_year = current_end_date.year
    valid_start = pd.Timestamp(f"{valid_year}-01-01")
    valid_end = pd.Timestamp(f"{valid_year}-12-31")

    mask_valid = (
        (valid_stocks["start_date"] <= valid_end) &
        (valid_stocks["end_date"] >= valid_start)
    )
    valid_subset = valid_stocks.loc[mask_valid].copy()

    # === 評估 coverage ===
    for df in [train_subset, valid_subset]:
        df["coverage_score"] = (
            (1 - df["missing_ratio_ipo"]) *
            (1 - df["zero_volume_ratio_file"]) * 100
        )
        df.sort_values("file", inplace=True)

    # === 輸出 Train / Valid 股票清單 ===
    pretrain_dir = OUTPUT_DIR / "incremental_pretrain"
    (pretrain_dir / "train").mkdir(exist_ok=True, parents=True)
    (pretrain_dir / "valid").mkdir(exist_ok=True, parents=True)

    train_file = pretrain_dir / "train" / f"{current_end_date.year}.csv"
    valid_file = pretrain_dir / "valid" / f"{current_end_date.year}.csv"

    train_subset[["file", "start_date", "end_date", "coverage_score"]].to_csv(train_file, index=False)
    valid_subset[["file", "start_date", "end_date", "coverage_score"]].to_csv(valid_file, index=False)

    print(f"🟢 Train {current_start_date.year}-{current_end_date.year} → {len(train_subset)} 檔")
    print(f"🟡 Valid {valid_year} → {len(valid_subset)} 檔")

    # === Online Fine-tuning ===
    fine_tune_year = current_end_date.year + 1
    ft_start = pd.Timestamp(f"{fine_tune_year}-01-01")
    ft_end = min(pd.Timestamp(f"{fine_tune_year}-12-31"), global_end_date)

    trading_days = calendar_df.query("(@ft_start <= k_datetime <= @ft_end)")["k_datetime"].tolist()
    if not trading_days:
        print(f"⚠️ 無 {fine_tune_year} 年交易日，跳過 Online FT")
        current_end_date += pd.DateOffset(years=STEP_YEARS)
        continue

    ft_dir = OUTPUT_DIR / f"online_ft/{fine_tune_year}"
    ft_dir.mkdir(exist_ok=True, parents=True)

    daily_summary = []
    for day in trading_days:
        mask_day = (valid_stocks["start_date"] <= day) & (valid_stocks["end_date"] >= day)
        day_stocks = valid_stocks.loc[mask_day, ["file", "start_date", "end_date"]].copy()
        day_stocks.sort_values("file", inplace=True)
        new_listed_count = (day_stocks["start_date"] == day).sum()
        day_stocks.to_csv(ft_dir / f"{day.date()}.csv", index=False)
        daily_summary.append({
            "date": day.date(),
            "num_stocks": len(day_stocks),
            "new_listed": new_listed_count
        })

    pd.DataFrame(daily_summary).to_csv(ft_dir / f"summary_{fine_tune_year}.csv", index=False)
    print(f"   🧾 Online FT {fine_tune_year}: 交易日 {len(trading_days)} 天完成")

    current_end_date += pd.DateOffset(years=STEP_YEARS)

print("\n✅ 已完成 Incremental Pretrain (train + valid) + Online Fine-tuning 股票池生成。")
