# =========================================================
# train_seq_transformer.py  (最終完整可重建版本)
# =========================================================
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn

from prepare_model_inputs import prepare_datasets
from sequence_transformer_model import (
    SeqStockTransformer,
    encode_calendar_dates,
)

# =========================================================
# 0. Command line
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

# =========================================================
# 1. 設定
# =========================================================
USE_CPU_ONLY = True
CPU_THREADS = 16
EPOCHS = 10
LR = 1e-3

HIDDEN_DIM = 256
NHEAD = 8
NUM_LAYERS = 4
SYM_EMB_DIM = 16
YEAR_EMB_DIM = 8

SAVE_DIR = "./checkpoints_seq"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.set_num_threads(CPU_THREADS)
DEVICE = "cuda" if torch.cuda.is_available() and not USE_CPU_ONLY else "cpu"
print("DEVICE =", DEVICE)


# =========================================================
# 2. 準備資料
# =========================================================
(
    train_ds, train_loader,
    valid_ds, valid_loader,
    test_ds, test_loader,
    symbols, dates
) = prepare_datasets()

first_batch = next(iter(train_loader))
feature_dim = first_batch["features"].shape[-1]
seq_len = first_batch["features"].shape[1]

print("[Data] feature_dim=", feature_dim, "seq_len=", seq_len)


# =========================================================
# 3. 建 vocab
# =========================================================
symbol2id = {s: i for i, s in enumerate(symbols)}
print("[Vocab] #symbols =", len(symbol2id))

year_list = sorted({int(str(d)[:4]) for d in dates})
year2id = {y: i for i, y in enumerate(year_list)}
print("[Vocab] #years =", len(year2id))



# =========================================================
# 4. Helper encoders
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
# 5. 建立模型
# =========================================================
model = SeqStockTransformer(
    feature_dim=feature_dim,
    hidden_dim=HIDDEN_DIM,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    num_symbols=len(symbol2id),
    sym_emb_dim=SYM_EMB_DIM,
    num_years=len(year2id),
    year_emb_dim=YEAR_EMB_DIM,
    max_len=seq_len,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()


print("Model params =", sum(p.numel() for p in model.parameters()) / 1e6, "M")


# =========================================================
# 6. Resume Training
# =========================================================
best_val = float("inf")
best_ckpt = None

train_losses = []
valid_losses = []
start_epoch = 1

if args.resume is not None:
    print(f"🔄 Resuming from checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if "train_losses" in ckpt:
        train_losses = ckpt["train_losses"]
        valid_losses = ckpt["valid_losses"]

    start_epoch = ckpt["epoch"] + 1
    print(f"➡️ Resume start epoch = {start_epoch}")

# =========================================================
# 7. 訓練迴圈
# =========================================================
for epoch in range(start_epoch, EPOCHS + 1):

    # ---------------- Train ----------------
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Train", ncols=100):

        x = batch["features"].to(DEVICE)
        y = batch["label"].to(DEVICE).squeeze(-1)

        sym_ids  = encode_symbols(batch["symbol"]).to(DEVICE)
        year_ids = encode_year_ids(batch["dates"]).to(DEVICE)
        cal_seq  = encode_calendar_seq(batch["dates"]).to(DEVICE)

        optimizer.zero_grad()
        pred = model(x, sym_ids, year_ids, cal_seq)

        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item() * x.size(0)

    avg_train = total_train_loss / len(train_loader.dataset)

    # ---------------- Valid ----------------
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"[Epoch {epoch}] Valid", ncols=100):

            x = batch["features"].to(DEVICE)
            y = batch["label"].to(DEVICE).squeeze(-1)

            sym_ids  = encode_symbols(batch["symbol"]).to(DEVICE)
            year_ids = encode_year_ids(batch["dates"]).to(DEVICE)
            cal_seq  = encode_calendar_seq(batch["dates"]).to(DEVICE)

            pred = model(x, sym_ids, year_ids, cal_seq)
            loss = criterion(pred, y)
            total_val_loss += loss.item() * x.size(0)

    avg_val = total_val_loss / len(valid_loader.dataset)
    print(f"Epoch {epoch} | Train={avg_train:.6f} | Valid={avg_val:.6f}")

    train_losses.append(avg_train)
    valid_losses.append(avg_val)


    # =========================================================
    # Save checkpoint
    # =========================================================
    ckpt_path = os.path.join(SAVE_DIR, f"seq_transformer_epoch{epoch:02d}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),

        "epoch": epoch,
        "train_loss": avg_train,
        "val_loss": avg_val,

        "train_losses": train_losses,
        "valid_losses": valid_losses,

        "symbol2id": symbol2id,
        "year2id": year2id,

        "config": {
            "feature_dim": feature_dim,
            "hidden_dim": HIDDEN_DIM,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "sym_emb_dim": SYM_EMB_DIM,
            "year_emb_dim": YEAR_EMB_DIM,
            "seq_len": seq_len,
            "num_symbols": len(symbol2id),
            "num_years": len(year2id),
        },
    }, ckpt_path)

    print("Saved:", ckpt_path)

    if avg_val < best_val:
        best_val = avg_val
        best_ckpt = ckpt_path

print("Done. Best ckpt =", best_ckpt)