"""
✅ Step 2：merge_features_v5.py（最終整合 + 可視化版）
-------------------------------------------------------
合併 incremental_pretrain (train/valid) 與 test 特徵，
生成統一格式 parquet，並輸出完整缺值 / 完整率統計與圖表。
-------------------------------------------------------
輸出：
  📦 merged_features.parquet
  📊 feature_coverage.csv
  📊 symbol_completeness.csv
  📊 symbol_date_completeness.parquet
  📄 completeness_summary.csv
  📝 merge_report.txt
  🖼️ symbol_completeness_hist.png
  🖼️ feature_coverage_bar.png
  🖼️ daily_completeness_trend.png
-------------------------------------------------------
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# === 路徑設定 ===
BASE_DIR = Path("./data/qlib_data/day1/generated_datasets")
OUTPUT_FILE = BASE_DIR / "merged_features.parquet"
REPORT_FILE = BASE_DIR / "merge_report.txt"

# === 幫助函數 ===
def load_and_tag_csv(file_path: Path, phase: str, year=None):
    """讀取單一 CSV 並加上 phase/year 標籤"""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if "symbol" not in df.columns or "date" not in df.columns:
            print(f"⚠️ {file_path.name} 缺少 symbol/date 欄位，跳過")
            return None
        df["phase"] = phase
        df["year"] = year or str(file_path.stem[:4])
        return df
    except Exception as e:
        print(f"❌ 無法讀取 {file_path.name}: {e}")
        return None


def collect_all_phases():
    """收集 incremental_pretrain (train/valid) + test（online_ft）"""
    all_files = []

    # 1) incremental_pretrain: train / valid
    for sub in ["train", "valid"]:
        phase_path = BASE_DIR / f"incremental_pretrain/{sub}"
        for f in sorted(phase_path.glob("*.csv")):
            if f.exists():
                df = load_and_tag_csv(f, phase=sub, year=f.stem)
                if df is not None:
                    all_files.append(df)

    # 2) online_ft，視為 test
    ft_base = BASE_DIR / "online_ft"
    for year_dir in sorted(ft_base.glob("*")):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.glob("*.csv")):
            df = load_and_tag_csv(f, phase="test", year=year_dir.name)
            if df is not None:
                all_files.append(df)

    return all_files

def detect_global_warmup(merged_df, feature_cols, threshold=0.5, min_days=10):
    """
    偵測全體股票前期 (warm-up) 應刪除的 row 數。
    - threshold: 平均完整率門檻 (例如 <0.5)
    - min_days: 連續幾天都達標才視為穩定期開始
    """
    print("📅 計算每日全體股票平均完整率 ...")

    daily_completeness = (
        merged_df.groupby("date")[feature_cols]
        .apply(lambda x: 1 - x.isna().mean().mean())
        .rename("avg_completeness")
        .reset_index()
    )

    # 找出連續 min_days 天完整率 >= threshold 的第一個位置
    consecutive_valid = 0
    cutoff_idx = 0
    for i, comp in enumerate(daily_completeness["avg_completeness"]):
        if comp >= threshold:
            consecutive_valid += 1
            if consecutive_valid >= min_days:
                cutoff_idx = i - min_days + 1
                break
        else:
            consecutive_valid = 0

    # 取得應刪除的日期區間
    if cutoff_idx == 0:
        print("⚠️ 沒找到連續達標區段，請降低 threshold 或 min_days。")
        return 0, daily_completeness

    cutoff_date = daily_completeness.iloc[cutoff_idx]["date"]
    print(f"✅ 全體股票完整率穩定起始日：{cutoff_date.strftime('%Y-%m-%d')}, index: {cutoff_idx}"
          f"(threshold={threshold}, min_days={min_days})")

    daily_completeness.to_csv(BASE_DIR / "daily_completeness.csv", index=False)
    print(f"📝 已輸出每日完整率 → {BASE_DIR / 'daily_completeness.csv'}")

    return cutoff_date, daily_completeness



# === 主流程 ===
def main():
    print("🧩 收集 incremental_pretrain / online_ft 特徵檔 ...")
    dfs = collect_all_phases()
    if not dfs:
        print("❌ 找不到任何特徵檔，請確認資料生成是否完成。")
        return

    # === 檢查欄位一致性 ===
    print("🔧 檢查欄位一致性 ...")
    common_cols = list(set.intersection(*(set(df.columns) for df in dfs)))
    merged_df = pd.concat([df[common_cols] for df in dfs], ignore_index=True)
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df = merged_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # === 特徵欄位 ===
    feature_cols = [
        c for c in merged_df.columns
        if c not in ["symbol", "date", "phase", "year"]
        and pd.api.types.is_numeric_dtype(merged_df[c])
    ]
    n_features = len(feature_cols)
    print(f"✅ 有效特徵欄位數：{n_features}")

    cutoff_date, completeness_df = detect_global_warmup(merged_df, feature_cols, threshold=0.99, min_days=10)
    if cutoff_date:
        merged_df = merged_df[merged_df["date"] >= cutoff_date].reset_index(drop=True)
        merged_df.to_parquet(BASE_DIR / "merged_features.parquet", index=False)
        print(f"✅ 已移除全體 warm-up 段（{cutoff_date.date()} 之前），合併，共 {len(merged_df):,} 筆 → merged_features.parquet")

    # === 欄位完整率 (Feature Coverage)
    feature_coverage = (1 - merged_df[feature_cols].isna().mean()).sort_values(ascending=False)
    feature_coverage.to_csv(BASE_DIR / "feature_coverage.csv")

    # === 股票完整率 (Symbol completeness)
    symbol_completeness = (
        merged_df.groupby("symbol")[feature_cols]
        .apply(lambda df: 1 - df.isna().mean().mean())
        .rename("symbol_completeness")
        .sort_values(ascending=False)
    )
    symbol_completeness.to_csv(BASE_DIR / "symbol_completeness.csv")

    # === 股票 × 日期完整率 (Symbol-Date completeness)
    symbol_date_completeness = (
        merged_df.groupby(["symbol", "date"])[feature_cols]
        .apply(lambda df: 1 - df.isna().mean(axis=1).iloc[0])
        .reset_index(name="completeness_ratio")
    )

    # === 每日整體完整率
    daily_completeness = (
        symbol_date_completeness.groupby("date")["completeness_ratio"]
        .mean()
        .rename("avg_completeness")
        .reset_index()
    )

    # === 統計摘要
    summary = {
        "n_rows": len(merged_df),
        "n_symbols": merged_df["symbol"].nunique(),
        "n_features": n_features,
        "avg_feature_coverage": feature_coverage.mean(),
        "avg_symbol_completeness": symbol_completeness.mean(),
        "min_symbol_completeness": symbol_completeness.min(),
        "max_symbol_completeness": symbol_completeness.max(),
        "avg_daily_completeness": daily_completeness["avg_completeness"].mean(),
    }
    pd.Series(summary).to_csv(BASE_DIR / "completeness_summary.csv")

    # === 報告輸出 ===
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("📘 Merge Features Report (v5 可視化版)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"資料筆數：{summary['n_rows']:,}\n")
        f.write(f"股票數量：{summary['n_symbols']}\n")
        f.write(f"特徵數量：{summary['n_features']}\n\n")
        f.write(f"平均特徵覆蓋率：{summary['avg_feature_coverage']:.4f}\n")
        f.write(f"平均股票完整率：{summary['avg_symbol_completeness']:.4f}\n")
        f.write(f"每日平均完整率：{summary['avg_daily_completeness']:.4f}\n")
        f.write(f"最小股票完整率：{summary['min_symbol_completeness']:.4f}\n")
        f.write(f"最大股票完整率：{summary['max_symbol_completeness']:.4f}\n")

    # === 可視化部分 ===
    print("🎨 產生圖表 ...")

    # 1️⃣ 每檔股票完整率分佈
    plt.figure(figsize=(8, 5))
    plt.hist(symbol_completeness, bins=50, edgecolor='black')
    plt.title("Symbol Completeness Distribution")
    plt.xlabel("Completeness Ratio")
    plt.ylabel("Number of Stocks")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "symbol_completeness_hist.png")
    plt.close()

    # 2️⃣ 特徵覆蓋率前後 20 名
    plt.figure(figsize=(10, 6))
    top_features = pd.concat([feature_coverage.head(10), feature_coverage.tail(10)])
    top_features.plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title("Feature Coverage (Top & Bottom 10)")
    plt.xlabel("Coverage Ratio")
    plt.ylabel("Feature Name")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "feature_coverage_bar.png")
    plt.close()

    # 3️⃣ 每日整體完整率趨勢
    plt.figure(figsize=(10, 5))
    plt.plot(daily_completeness["date"], daily_completeness["avg_completeness"], lw=1.5)
    plt.title("Daily Average Completeness Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Completeness")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "daily_completeness_trend.png")
    plt.close()

    # === 輸出摘要 ===
    print("📊 統計與圖表已輸出至資料夾：")
    for file in [
        "feature_coverage.csv", "symbol_completeness.csv",
        "symbol_date_completeness.parquet", "completeness_summary.csv",
        "symbol_completeness_hist.png", "feature_coverage_bar.png", "daily_completeness_trend.png"
    ]:
        print(" └─", BASE_DIR / file)


if __name__ == "__main__":
    main()
