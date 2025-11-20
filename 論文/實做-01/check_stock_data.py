"""
===============================================================
📘 股票歷史資料完整性檢查工具 (v2 Clean Edition)
---------------------------------------------------------------
用途：
    - 檢查多檔股票歷史資料（每檔一個 CSV）
    - 以加權指數或市場日曆 (weight.csv) 為基準
    - 確認每支股票的日期對齊、缺值、停牌天數、零量比例等
    - 自動生成報告 data_check_summary.csv

特點：
    ✅ 以股票掛牌期間 (IPO window) 為基準計算缺值與停牌
    ✅ 將全市場基準 (full window) 指標保留作參考
    ✅ 僅使用實際檔案內 volume 計算停牌比例
    ✅ 明確區分錯誤、警示、與正常股票

使用方式：
    python check_with_calendar_v2_clean.py

檔案需求：
    1. weigt.csv (含所有交易日，欄位 k_datetime)
    2. data/day1/*.csv (個股歷史檔，欄位至少含 k_datetime, open, high, low, close, volume)

輸出：
    data_check_summary.csv


===============================================================
"""

import os
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------
# 🧩 參數設定
# ---------------------------------------------------------------
CALENDAR_CSV = "data/day1/weigt.csv"        # 含所有交易日 (k_datetime)
STOCK_DIR = "data/day1"            # 個股資料目錄
SUMMARY_OUT = "data/day1/summary_data.csv"

# 警示閾值設定
LONG_GAP_WARN_DAYS = 7             # 連續缺日超過 7 天警告
LONG_GAP_EXCL_DAYS = 30            # 超過 30 天通常剔除
ZERO_VOL_WARN_RATIO = 0.5          # volume=0 比例超過 50% 警告
ZERO_VOL_STREAK_WARN = 10          # 連續 volume=0 超過 10 天警告


# ---------------------------------------------------------------
# 🧠 工具函式
# ---------------------------------------------------------------

def longest_streak(bool_series: pd.Series) -> int:
    """計算布林序列中 True 的最長連續長度。"""
    max_len = 0
    count = 0
    for value in bool_series:
        if value:
            count += 1
            max_len = max(max_len, count)
        else:
            count = 0
    return max_len


# ---------------------------------------------------------------
# ⚙️ 主程式
# ---------------------------------------------------------------
def run():
    # === 1️⃣ 讀取市場交易日曆 ===
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["k_datetime"])
    cal = cal.sort_values("k_datetime")[["k_datetime"]].drop_duplicates().reset_index(drop=True)
    cal_dates = cal["k_datetime"]

    results = []
    stock_files = [f for f in os.listdir(STOCK_DIR) if f.endswith(".csv")]

    # === 2️⃣ 逐檔檢查個股資料 ===
    for fname in tqdm(stock_files):
        path = os.path.join(STOCK_DIR, fname)
        result = {"file": fname}

        try:
            df = pd.read_csv(path, parse_dates=["k_datetime"])
        except Exception as e:
            result.update({"error": f"read_error:{e}"})
            results.append(result)
            continue

        # 檢查必要欄位
        required_cols = {"k_datetime", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            result.update({"error": "missing_columns"})
            results.append(result)
            continue

        # 排序日期與去重
        df = df.sort_values("k_datetime").drop_duplicates(subset=["k_datetime"]).reset_index(drop=True)
        if df.empty:
            result.update({"error": "empty_file"})
            results.append(result)
            continue

        # === 基本資訊 ===
        start_date = df["k_datetime"].iloc[0]
        end_date = df["k_datetime"].iloc[-1]
        result.update({
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "row_count": len(df)
        })

        # === 3️⃣ 是否有非交易日紀錄 ===
        has_non_trading = (~df["k_datetime"].isin(cal_dates)).any()
        result["has_non_trading_dates"] = int(has_non_trading)

        # === 4️⃣ 全市場日曆基準 (full window) ===
        merged_full = cal.merge(df, on="k_datetime", how="left", sort=True)
        is_missing_full = merged_full[["open", "high", "low", "close", "volume"]].isna().all(axis=1)
        result["missing_days_full"] = int(is_missing_full.sum())
        result["missing_ratio_full"] = round(float(is_missing_full.mean()), 6)
        result["longest_missing_streak_full"] = longest_streak(is_missing_full)

        # === 5️⃣ 掛牌期間基準 (IPO window) ===
        cal_ipo = cal[(cal["k_datetime"] >= start_date) & (cal["k_datetime"] <= end_date)]
        merged_ipo = cal_ipo.merge(df, on="k_datetime", how="left", sort=True)
        is_missing_ipo = merged_ipo[["open", "high", "low", "close", "volume"]].isna().all(axis=1)

        result["missing_days_ipo"] = int(is_missing_ipo.sum())
        result["missing_ratio_ipo"] = round(float(is_missing_ipo.mean()), 6)
        result["longest_missing_streak_ipo"] = longest_streak(is_missing_ipo)

        # === 6️⃣ 檔案內日期間隔 ===
        if len(df) >= 2:
            result["max_gap_by_diff"] = int(df["k_datetime"].diff().dt.days.dropna().max())
        else:
            result["max_gap_by_diff"] = 0

        # === 7️⃣ 檔案內零量統計 ===
        zero_vol = df["volume"].fillna(0).eq(0)
        result["zero_vol"] = zero_vol.sum()
        result["zero_volume_ratio_file"] = round(float(zero_vol.mean()), 6)
        result["longest_zero_vol_streak_file"] = longest_streak(zero_vol)

        # === 8️⃣ 警示旗標 ===
        warn_flags = []
        if has_non_trading:
            warn_flags.append("has_non_trading_dates")
        if result["longest_missing_streak_ipo"] >= LONG_GAP_WARN_DAYS:
            warn_flags.append(f"gap_ipo>={LONG_GAP_WARN_DAYS}")
        if result["longest_missing_streak_ipo"] >= LONG_GAP_EXCL_DAYS:
            warn_flags.append(f"gap_ipo>={LONG_GAP_EXCL_DAYS}")
        if result["zero_volume_ratio_file"] > ZERO_VOL_WARN_RATIO:
            warn_flags.append(f"zero_vol>={ZERO_VOL_WARN_RATIO*100:.0f}%")
        if result["longest_zero_vol_streak_file"] >= ZERO_VOL_STREAK_WARN:
            warn_flags.append(f"zero_vol_streak>={ZERO_VOL_STREAK_WARN}")

        result["warn_flag"] = "|".join(warn_flags) if warn_flags else ""
        result["error"] = ""  # 成功

        results.append(result)

    # === 9️⃣ 輸出報告 ===
    report = pd.DataFrame(results)
    report.to_csv(SUMMARY_OUT, index=False)
    print(f"✅ 完成！共 {len(report)} 檔股票，報告輸出至：{SUMMARY_OUT}")


# ---------------------------------------------------------------
# 🚀 執行
# ---------------------------------------------------------------
if __name__ == "__main__":
    run()
