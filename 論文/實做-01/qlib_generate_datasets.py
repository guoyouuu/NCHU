"""
📘 qlib_genrate_datasets.py
-------------------------------------------------------
依據 Incremental Pretraining / Online Fine-tuning 階段，
讀取股票池定義檔 (CSV)，
自動產生訓練或測試資料集（Qlib 可用格式）。

使用方式：
-------------------------------------------------------
Incremental Pretraining：
python ./src/qlib_generate_datasets.py --mode incremental_pretrain --year 1992

Online Fine-tuning：
# python ./src/qlib_generate_datasets.py --mode online_ft --date 1993-01-05
python ./src/qlib_generate_datasets.py --mode online_ft --year 1993
-------------------------------------------------------
"""

import argparse
import pandas as pd
from pathlib import Path
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import qlib

# === 初始化 Qlib ===
qlib.init(provider_uri="./data/qlib_data/day1", region="cn", num_workers=8)
print("✅ Qlib Initialized")

# === 參數設定 ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["incremental_pretrain", "online_ft"], required=True)
parser.add_argument("--year", type=int, help="Incremental pretraining year (e.g. 1992)")
parser.add_argument("--date", type=str, help="Online fine-tuning date (e.g. 1993-01-05)")
args = parser.parse_args()

# === 路徑設定 ===
BASE_DIR = Path("./data/stock_pools")
PRETRAIN_DIR = BASE_DIR / "incremental_pretrain"
ONLINE_FT_DIR = BASE_DIR / "online_ft"
OUTPUT_DIR = Path("./data/qlib_data/day1/generated_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 共用函式：產生 Alpha158 特徵 ===
def generate_alpha158_features(symbols, start_time, end_time, fit_start_time, fit_end_time, output_path):
    """產生 Alpha158 特徵並輸出 CSV"""
    print(f"🧮 Generating Alpha158 features for {len(symbols)} stocks: {start_time} → {end_time}")

    handler = Alpha158(
        instruments=symbols,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
    )

    # dataset = DatasetH(handler, segments={"full": (start_time, end_time)})

    # 取得特徵名稱
    ohlcv_fields = ["$open", "$high", "$low", "$close", "$volume"]
    ohlcv_names = ["open", "high", "low", "close", "volume"]
    feature_expressions, feature_names = handler.get_feature_config()
    df = D.features(symbols, ohlcv_fields + feature_expressions, start_time=start_time, end_time=end_time)
    
    if df.empty:
        print(f"⚠️ {output_path} 為空（可能該日無交易或 symbol 不符）")
    else:
        expected_cols = len(ohlcv_names) + len(feature_names)
        if df.shape[1] != expected_cols:
            print(f"⚠️ 欄位數量不符：預期 {expected_cols}，實際 {df.shape[1]}")
        df.columns = ohlcv_names + feature_names


    # 換成可讀名稱
    df.columns = ohlcv_names + feature_names

    # 展平成標準表格
    df = df.reset_index().rename(columns={"datetime": "date", "instrument": "symbol"})

    df.to_csv(output_path, index=False)
    print(f"✅ Saved Alpha158 dataset → {output_path}")

# === 模式 A：Incremental Pretraining ===
if args.mode == "incremental_pretrain":
    # === 分別讀取 train / valid 股票池 ===
    train_pool_path = BASE_DIR / f"incremental_pretrain/train/{args.year}.csv"
    valid_pool_path = BASE_DIR / f"incremental_pretrain/valid/{args.year}.csv"

    if not train_pool_path.exists() or not valid_pool_path.exists():
        raise FileNotFoundError(f"❌ 找不到股票池檔案：{train_pool_path} 或 {valid_pool_path}")

    train_pool = pd.read_csv(train_pool_path)
    valid_pool = pd.read_csv(valid_pool_path)
    print(f"📘 Loaded Incremental Pretrain TRAIN ({args.year}) — {len(train_pool)} stocks")
    print(f"📘 Loaded Incremental Pretrain VALID ({args.year}) — {len(valid_pool)} stocks")

    train_symbols = [Path(f).stem for f in train_pool["file"]]
    valid_symbols = [Path(f).stem for f in valid_pool["file"]]
    print(f"Train symbols: {train_symbols[:5]} ... Total: {len(train_symbols)}")
    print(f"Valid symbols: {valid_symbols[:5]} ... Total: {len(valid_symbols)}")

    # === 定義時間區間 ===
    train_start = train_pool["start_date"].min()
    train_end = f"{args.year - 1}-12-31"
    valid_start = f"{args.year}-01-01"
    valid_end = f"{args.year}-12-31"

    # === 產生 train/valid 特徵 ===
    train_out = OUTPUT_DIR / f"incremental_pretrain/train/{args.year}.csv"
    valid_out = OUTPUT_DIR / f"incremental_pretrain/valid/{args.year}.csv"
    train_out.parent.mkdir(parents=True, exist_ok=True)
    valid_out.parent.mkdir(parents=True, exist_ok=True)

    generate_alpha158_features(
        train_symbols,
        start_time=train_start,
        end_time=train_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        output_path=train_out
    )
    generate_alpha158_features(
        valid_symbols,
        start_time=valid_start,
        end_time=valid_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        output_path=valid_out
    )


# # === 模式 B：Online Fine-tuning ===
# elif args.mode == "online_ft":
#     year = args.date[:4]
#     csv_path = ONLINE_FT_DIR / year / f"{args.date}.csv"

#     if not csv_path.exists():
#         raise FileNotFoundError(f"❌ 找不到股票池檔案：{csv_path}")
    
#     stock_pool = pd.read_csv(csv_path)
#     print(f"📘 Loaded Online FT Universe ({args.date}) — {len(stock_pool)} stocks")

#     symbols = [Path(f).stem for f in stock_pool["file"]]
#     print(f"Symbols: {symbols[:5]} ... Total: {len(symbols)}")

#     # 單日（或可改成 rolling 多日）
#     train_start = stock_pool["start_date"].min()
#     train_end = args.date
#     test_start = args.date
#     test_end = args.date

#     output_path = OUTPUT_DIR / f"online_ft/{year}/{args.date}.csv"
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     generate_alpha158_features(
#         symbols, 
#         start_time=test_start,
#         end_time=test_end,
#         fit_start_time=train_start,
#         fit_end_time=train_end, 
#         output_path=output_path
#     )

# === 模式 B：Online Fine-tuning（改為輸入 year，生成該年所有日期） ===
elif args.mode == "online_ft":

    if args.year is None:
        raise ValueError("❌ online_ft 模式必須要指定 --year，例如 --year 1993")

    year = args.year
    pool_dir = ONLINE_FT_DIR / str(year)

    if not pool_dir.exists():
        raise FileNotFoundError(f"❌ 找不到股票池資料夾：{pool_dir}")

    # 收集該年度所有池文件（每天一個 CSV）
    csv_files = sorted(pool_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"❌ 沒有找到任何股票池 CSV：{pool_dir}")

    print(f"📘 Loaded Online FT Universe YEAR={year} — {len(csv_files)} days")

    # 逐日產生 Alpha158 特徵
    for csv_path in csv_files:
        date_str = csv_path.stem   # 1993-01-05
        stock_pool = pd.read_csv(csv_path)

        symbols = [Path(f).stem for f in stock_pool["file"]]
        print(f"\n📅 {date_str} — {len(symbols)} symbols")

        train_start = stock_pool["start_date"].min()
        train_end = date_str
        test_start = date_str
        test_end = date_str   # 單日特徵

        output_path = OUTPUT_DIR / f"online_ft/{year}/{date_str}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generate_alpha158_features(
            symbols,
            start_time=test_start,
            end_time=test_end,
            fit_start_time=train_start,
            fit_end_time=train_end,
            output_path=output_path,
        )
else:
    raise ValueError("❌ Mode must be 'incremental_pretrain' or 'online_ft'")
