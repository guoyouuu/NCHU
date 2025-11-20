# =========================================================
# sequence_transformer_model.py  (修正版)
# =========================================================
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------
# Positional Encoding：對序列位置 (0~L-1) 做 sin/cos
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------
# calendar sin/cos encoding（不含 year → 年份改 embedding）
# ---------------------------------------------------------
def encode_calendar_dates(date_list):
    """
    date_list: list[str]，flatten 後的日期
    回傳: [N, 6]（month/day/weekday 的 sin/cos）
    """
    dt = pd.to_datetime(pd.Series(date_list))

    month = dt.dt.month.to_numpy().astype(np.float32)     # 1~12
    day   = dt.dt.day.to_numpy().astype(np.float32)       # 1~31
    wday  = dt.dt.weekday.to_numpy().astype(np.float32)   # 0~6

    # sin/cos encoding
    def cyc(x, period):
        return np.sin(2*np.pi*x/period), np.cos(2*np.pi*x/period)

    sm, cm = cyc(month, 12)
    sd, cd = cyc(day, 31)
    sw, cw = cyc(wday, 7)

    out = np.stack([sm, cm, sd, cd, sw, cw], axis=1).astype("float32")
    return torch.from_numpy(out)     # [N, 6]


# ---------------------------------------------------------
# **SeqStockTransformer（最終正確版）**
#   - symbol embedding
#   - year embedding
#   - per-step calendar sin/cos
# ---------------------------------------------------------
class SeqStockTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
        nhead=8,
        num_layers=4,
        num_symbols=3000,
        sym_emb_dim=16,
        num_years=60,             # ⭐ 新增
        year_emb_dim=8,           # ⭐ 新增
        cal_dim=6,                # sin/cos(month/day/wday)
        max_len=512,
    ):
        super().__init__()

        # symbol categorical
        self.sym_emb = nn.Embedding(num_symbols, sym_emb_dim)

        # year categorical（regime embedding）
        self.year_emb = nn.Embedding(num_years, year_emb_dim)

        # concat: feature + sym_emb + year_emb + cal_sin_cos
        input_dim = feature_dim + sym_emb_dim + year_emb_dim + cal_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc_out = nn.Linear(hidden_dim, 1)

    # -----------------------------------------------------
    # forward（最終正確版）
    # -----------------------------------------------------
    def forward(self, x, sym_ids, year_ids, cal_seq):
        """
        x        : [B, L, F]
        sym_ids  : [B]
        year_ids : [B, L]
        cal_seq  : [B, L, 6]  per-step sin/cos dates
        """
        B, L, _ = x.shape

        # --- symbol embedding ---
        sym_vec = self.sym_emb(sym_ids)       # [B, sym_dim]
        sym_vec = sym_vec.unsqueeze(1).expand(B, L, -1)

        # --- year embedding (per step) ---
        year_vec = self.year_emb(year_ids)    # [B, L, year_emb_dim]

        # --- concat all inputs ---
        x = torch.cat([x, sym_vec, year_vec, cal_seq], dim=-1)

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        out = self.fc_out(x[:, -1])           # last time step
        return out.squeeze(-1)
