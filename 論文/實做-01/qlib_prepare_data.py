import pandas as pd
from pathlib import Path
from tqdm import tqdm


# === 1️⃣ 參數設定 ===
TIME_INTERVAL = "day1"  # 可改成 day1 min1 等其他時間頻率
DATA_DIR = Path(f"./data/{TIME_INTERVAL}")
PREPARE_DATA_DIR = Path(f"./data/qlib_prepare_data/{TIME_INTERVAL}")  # 儲存轉換後的資料
PREPARE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# === 2️⃣ 逐檔轉換成 Qlib 標準欄位 ===
for file in tqdm(sorted(DATA_DIR.glob("*.csv")), desc="轉換股票資料"):
    symbol = file.stem.lower()
    df = pd.read_csv(file, parse_dates=["k_datetime"])
    
    # 加入 symbol 欄位（Qlib 必須）
    df["symbol"] = symbol
    
    # 調整欄位順序
    df = df[["symbol", "k_datetime", "open", "high", "low", "close", "volume"]]
    
    # 排序確保時間序一致
    df = df.sort_values("k_datetime").reset_index(drop=True)
    
    # 存回新資料夾
    out_path = PREPARE_DATA_DIR / f"{symbol}.csv"
    df.to_csv(out_path, index=False)
    
    # print(f"✅ 已轉換：{symbol} → {out_path.name}")

print("\n🎉 全部股票已轉換完成，可直接用 Qlib 進行 dump。")

"""
python qlib/scripts/dump_bin.py dump_all --data_path ./data/qlib_prepare_data/day1 --qlib_dir ./data/qlib_data/day1 --freq day --date_field_name k_datetime --include_fields open,high,low,close,volume
"""