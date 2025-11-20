# """
# ✅ verify_alpha158_indicators.py
# -------------------------------------------------------
# 重現 Qlib Alpha158 所有技術指標並驗證結果一致性
# 使用方法：
#     python verify_alpha158_indicators.py --symbol 1101
# """
import ast
# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from qlib.contrib.data.handler import Alpha158
# import qlib
# import re

# # =======================================================
# # 1️⃣ 初始化 Qlib
# # =======================================================
# qlib.init(provider_uri="./data/qlib_data/day1", region="cn", num_workers=8)
# print("✅ Qlib Initialized")

# # =======================================================
# # 2️⃣ 運算函數表（完整擴充）
# # =======================================================
# OPS = {
#     # === 基本數學 ===
#     "Greater": np.maximum,
#     "Less": np.minimum,
#     "Abs": np.abs,
#     # "Log": np.log1p,
#     "Log": np.log,
#     "SignedPower": lambda x, y: np.sign(x) * np.abs(x) ** y,

#     # === 滾動統計 ===
#     "Mean": lambda x, n: x.rolling(int(n), min_periods=1).mean(),
#     "Std": lambda x, n: x.rolling(int(n), min_periods=1).std(),
#     "Var": lambda x, n: x.rolling(int(n), min_periods=1).var(),
#     "Max": lambda x, n: x.rolling(int(n), min_periods=1).max(),
#     "Min": lambda x, n: x.rolling(int(n), min_periods=1).min(),
#     "Sum": lambda x, n: x.rolling(int(n), min_periods=1).sum(),
#     "Delay": lambda x, n: x.shift(int(n)),
#     "Ref": lambda x, n: x.shift(int(n)),

#     # === Rank / Quantile ===
#     "Rank": lambda x, n=None: (
#         x.rolling(int(n), min_periods=1).apply(lambda s: s.rank(pct=True).iloc[-1])
#         if n else x.rank(pct=True)
#     ),
#     "Quantile": lambda x, n, q: x.rolling(int(n), min_periods=1).quantile(float(q)),

#     # === Regression 與統計 ===
#     "Slope": lambda x, n: x.rolling(int(n), min_periods=3).apply(
#         lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s.dropna()) > 2 else np.nan
#     ),
#     "Rsquare": lambda x, n: x.rolling(int(n), min_periods=3).apply(
#         lambda s: (np.corrcoef(np.arange(len(s)), s)[0, 1] ** 2)
#         if s.std() > 0 else np.nan
#     ),
#     "Resi": lambda x, n: x.rolling(int(n), min_periods=3).apply(
#         lambda s: s.iloc[-1] - np.poly1d(np.polyfit(np.arange(len(s)), s, 1))(len(s) - 1)
#         if len(s.dropna()) > 2 else np.nan
#     ),
#     "Beta": lambda x, y, n: x.rolling(int(n), min_periods=1).cov(y)
#         / (y.rolling(int(n), min_periods=1).var() + 1e-12),

#     # === IdxMax / IdxMin ===
#     "IdxMax": lambda x, n: x.rolling(int(n), min_periods=1).apply(
#         lambda s: np.argmax(s) + 1, raw=True
#     ),
#     "IdxMin": lambda x, n: x.rolling(int(n), min_periods=1).apply(
#         lambda s: np.argmin(s) + 1, raw=True
#     ),
#     "IMXD": lambda x, n: OPS["IdxMax"](x, n) - OPS["IdxMin"](x, n),

#     # === CNTP / CNTN / CNTD ===
#     "CNTP": lambda x, n: x.diff().fillna(0).gt(0).rolling(int(n), min_periods=1).sum(),
#     "CNTN": lambda x, n: x.diff().fillna(0).lt(0).rolling(int(n), min_periods=1).sum(),
#     "CNTD": lambda x, n: x.diff().fillna(0).eq(0).rolling(int(n), min_periods=1).sum(),

#     # === Volume Weighted Mean / Std ===
#     "VMA": lambda v, n: OPS["Mean"](v, n) / (v + 1e-12),
#     "VSTD": lambda v, n: OPS["Std"](v, n) / (v + 1e-12),

#     # === Corr / Cov ===
#     # "Corr": lambda x, y, n: x.rolling(int(n), min_periods=3).corr(y),
#     "Cov": lambda x, y, n: x.rolling(int(n), min_periods=3).cov(y),

#     # === Corr / Cord ===
#     "Corr": lambda x, y, n: x.rolling(int(n), min_periods=3).corr(y),
#     "CORD": lambda x, y, n: (
#         (x / OPS["Ref"](x, 1))
#         .rolling(int(n), min_periods=3)
#         .corr(np.log(y / OPS["Ref"](y, 1) + 1))
#     )
# }

# # =======================================================
# # 3️⃣ Qlib DSL → Python 可執行轉換
# # =======================================================
# def qlib_expr_to_python(expr: str) -> str:
#     expr = expr.strip()
#     expr = re.sub(r"\$(\w+)", r'env["\1"]', expr)
#     expr = re.sub(r"\b([A-Z][A-Za-z0-9_]*)\s*\(", r'OPS["\1"](', expr)
#     expr = re.sub(r"\s+", " ", expr)
#     return expr

# # =======================================================
# # 4️⃣ AST 安全解析
# # =======================================================
# class SafeEvaluator(ast.NodeVisitor):
#     def __init__(self, env):
#         self.env = env

#     def visit_Name(self, node):
#         if node.id == "env": return self.env
#         if node.id == "OPS": return OPS
#         if node.id in self.env: return self.env[node.id]
#         raise ValueError(f"未知名稱: {node.id}")

#     def visit_Constant(self, node): return node.value

#     def visit_BinOp(self, node):
#         left, right = self.visit(node.left), self.visit(node.right)
#         if isinstance(node.op, ast.Add): return left + right
#         if isinstance(node.op, ast.Sub): return left - right
#         if isinstance(node.op, ast.Mult): return left * right
#         # if isinstance(node.op, ast.Div): return left / (right + 1e-12)
#         if isinstance(node.op, ast.Div):  return left / right   # ← 不要偷偷 +1e-12
#         raise ValueError(f"Unsupported operator: {node.op}")

#     def visit_UnaryOp(self, node):
#         val = self.visit(node.operand)
#         return -val if isinstance(node.op, ast.USub) else val

#     def visit_Call(self, node):
#         func = self.visit(node.func)
#         args = [self.visit(a) for a in node.args]
#         return func(*args)

#     def visit_Attribute(self, node):
#         value = self.visit(node.value)
#         return getattr(value, node.attr)

#     def visit_Subscript(self, node):
#         value = self.visit(node.value)
#         key = self.visit(node.slice)
#         return value[key]

#     def visit_Compare(self, node):
#         left = self.visit(node.left)
#         right = self.visit(node.comparators[0])
#         op = node.ops[0]
#         if isinstance(op, ast.Gt):
#             return (left > right).astype(float)
#         elif isinstance(op, ast.Lt):
#             return (left < right).astype(float)
#         elif isinstance(op, ast.GtE):
#             return (left >= right).astype(float)
#         elif isinstance(op, ast.LtE):
#             return (left <= right).astype(float)
#         elif isinstance(op, ast.Eq):
#             return (left == right).astype(float)
#         elif isinstance(op, ast.NotEq):
#             return (left != right).astype(float)
#         else:
#             raise ValueError(f"Unsupported comparison operator: {op}")

# # =======================================================
# # 5️⃣ 主驗證流程
# # =======================================================
# def verify_symbol(symbol: str, year: int = 1992,
#                   train_type: str = "train",
#                   data_dir="./data/day1",
#                   result_dir="./data/qlib_data/day1/generated_datasets/incremental_pretrain"):
#     raw = pd.read_csv(Path(data_dir) / f"{symbol}.csv")
#     raw["datetime"] = pd.to_datetime(raw["k_datetime"])
#     raw.set_index("datetime", inplace=True)
#     raw = raw[~raw.index.duplicated(keep="first")]
#     raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce") 
#     raw["vwap"] = (raw["high"] + raw["low"] + raw["close"]) / 3

#     env = {c: raw[c] for c in ["open", "high", "low", "close", "volume", "vwap"]}
    
#     handler = Alpha158(instruments=[symbol], start_time="1980-01-01", end_time="2024-12-31")
#     feature_expressions, feature_names = handler.get_feature_config()

#     evaluator = SafeEvaluator(env)
#!     results = {}
#     for name, expr in zip(feature_names, feature_expressions):
#         try:
#             tree = ast.parse(qlib_expr_to_python(expr), mode="eval")
#             results[name] = evaluator.visit(tree.body)
#         except Exception as e:
#             results[name] = None
#!             print(f"{name} ❌ {e}")

#     # === 讀取 qlib 輸出結果 ===
#     result_path = Path(result_dir) / f"{train_type}/{year}.csv"    
#     qlib_result = pd.read_csv(result_path)
#     qlib_result["symbol"] = qlib_result["symbol"].astype(str).str.strip()
#     qlib_result["date"] = pd.to_datetime(qlib_result["date"])
#     qlib_result.set_index("date", inplace=True)
#     qlib_result = qlib_result[qlib_result["symbol"] == symbol].copy()
#     print(qlib_result.head())
    
#     out = []
#     compare_all = []  # 🧩 收集所有特徵對照表


#     for name in feature_names:
#         if results[name] is None or name not in qlib_result.columns:
#             out.append((name, None))
#             continue

        
#         # === 移除重複日期 ===
#         s1 = results[name].loc[~results[name].index.duplicated(keep="first")]
#         s2 = qlib_result[name].loc[~qlib_result.index.duplicated(keep="first")]

#         # === 對齊日期 ===
#         s1, s2 = s1.align(s2, join="inner")

#         if len(s1) == 0:
#             out.append((name, None))
#             continue

#         mae = np.nanmean(np.abs(s1 - s2))
#         out.append((name, mae))

#         # === 🧩 收集完整對照結果 ===
#         compare_df = pd.DataFrame({
#             "feature": name,
#             "date": s1.index,
#             "our_calc": s1.values,
#             "qlib_calc": s2.values,
#             "diff": (s1 - s2).values
#         })
#         compare_all.append(compare_df)

#     # === 統計結果 ===
#     out_df = pd.DataFrame(out, columns=["feature", "MAE"])
#     out_path = Path(f"./verify_result_{symbol}.csv")
#     out_df.to_csv(out_path, index=False)
#     print(f"\n✅ 指標誤差統計輸出：{out_path}")

#     # === 合併所有對照 ===
#     compare_all_df = pd.concat(compare_all, ignore_index=True)
#     compare_out_path = Path(f"./verify_result_compare_{symbol}.csv")
#     compare_all_df.to_csv(compare_out_path, index=False)
#     print(f"✅ 完整對照輸出：{compare_out_path}")


#     # === Step X: 找出誤差大的特徵進行 Debug ===
#     threshold = 1e-3  # 🔧 誤差門檻，可調整
#     print(f"\n=== 🔍 差異超過 {threshold} 的特徵 Debug ===")

#     # 載入結果
#     out_df = pd.read_csv(f"./verify_result_{symbol}.csv")
#     diff_features = out_df[out_df["MAE"].fillna(0) > threshold]["feature"].tolist()

#     # # 用相對誤差（推薦）
#     # out_df["relative_err"] = out_df["MAE"] / (out_df["MAE"].abs().mean() + 1e-12)
#     # diff_features = out_df[out_df["relative_err"] > 1e-6]["feature"].tolist()
    

#     if not diff_features:
#         print("✅ 所有特徵誤差皆在正常範圍內")
#     else:
#         print(f"⚠️ 共 {len(diff_features)} 個特徵超過閾值：{diff_features}")

#         compare_all_df = pd.read_csv(f"./verify_result_compare_{symbol}.csv")

#         # 重新載入 feature config
#         handler = Alpha158(instruments=[symbol], start_time="2010-01-01", end_time="2024-12-31")
#         feature_expressions, feature_names = handler.get_feature_config()
#         expr_map = dict(zip(feature_names, feature_expressions))

#         for name in diff_features:
#             expr = expr_map.get(name)
#             if expr is None:
#                 print(f"\n=== Debug {name} ===\n⚠️ 找不到公式")
#                 continue

#             print(f"\n=== Debug {name} ===")
#             print("原始公式:", expr)

#             try:
#                 py_expr = qlib_expr_to_python(expr)
#                 print("轉換後公式:", py_expr)

#                 tree = ast.parse(py_expr, mode="eval")
#                 our_calc = SafeEvaluator(env).visit(tree.body)

#                 # 取出 qlib 計算值
#                 sub = compare_all_df[compare_all_df["feature"] == name].copy()
#                 sub["abs_diff"] = sub["diff"].abs()
#                 sub_top = sub.sort_values("abs_diff", ascending=False).head(5)

#                 print(f"✅ Qlib vs 我們計算差異 Top 5:")
#                 print(sub_top[["date", "our_calc", "qlib_calc", "diff"]])
#             except Exception as e:
#                 print(f"{name} ❌ 錯誤:", e)


# # =======================================================
# # 6️⃣ 執行入口
# # =======================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--symbol", required=True, help="股票代號，例如 1101")
#     parser.add_argument("--train_type", default="train", help="train / valid")
#     args = parser.parse_args()
#     verify_symbol(args.symbol, train_type=args.train_type)

    

"""
✅ qlib_verify_genrate_datasets.py
-------------------------------------------------------
重現 Qlib Alpha158 所有技術指標並驗證結果一致性

支援模式：
1️⃣ incremental_pretrain（會同時驗證 train + valid）
2️⃣ online_ft（驗證指定日期）

使用方式：
-------------------------------------------------------
Incremental Pretrain 驗證：
python ./src/qlib_verify_genrate_datasets.py --symbol 1101 --mode incremental_pretrain --year 1992

Online Fine-tuning 驗證：
python ./src/qlib_verify_genrate_datasets.py --symbol 1101 --mode online_ft --date 1993-01-05
-------------------------------------------------------
"""

import ast
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158
import qlib
import re

# =======================================================
# 1️⃣ 初始化 Qlib
# =======================================================
qlib.init(provider_uri="./data/qlib_data/day1", region="cn", num_workers=8)
print("✅ Qlib Initialized")

# =======================================================
# 2️⃣ 運算函數表（OPS）
# =======================================================
OPS = {
    # ---------------------------------------------------
    # 基本運算類（單變數操作）
    # ---------------------------------------------------
    "Greater": np.maximum,     # 取較大值
    "Less": np.minimum,        # 取較小值
    "Abs": np.abs,             # 絕對值
    "Log": np.log,             # 自然對數
    "SignedPower": lambda x, y: np.sign(x) * np.abs(x) ** y,  # 保留符號的次方

    # ---------------------------------------------------
    # 滾動統計類（Rolling Statistics）
    # ---------------------------------------------------
    "Mean": lambda x, n: x.rolling(int(n), min_periods=1).mean(),  # 滾動平均
    "Std":  lambda x, n: x.rolling(int(n), min_periods=1).std(),   # 滾動標準差
    "Var":  lambda x, n: x.rolling(int(n), min_periods=1).var(),   # 滾動變異數
    "Max":  lambda x, n: x.rolling(int(n), min_periods=1).max(),   # 滾動最大值
    "Min":  lambda x, n: x.rolling(int(n), min_periods=1).min(),   # 滾動最小值
    "Sum":  lambda x, n: x.rolling(int(n), min_periods=1).sum(),   # 滾動加總
    "Ref":  lambda x, n: x.shift(int(n)),                          # 往前位移（Ref/Delay 等價）

    # ---------------------------------------------------
    # 排名與分位類（Ranking & Quantile）
    # ---------------------------------------------------
    "Rank": lambda x, n=None: (  # 區間內百分比排名
        x.rolling(int(n), min_periods=1).apply(lambda s: s.rank(pct=True).iloc[-1])
        if n else x.rank(pct=True)
    ),
    "Quantile": lambda x, n, q: x.rolling(int(n), min_periods=1).quantile(float(q)),  # 分位數

    # ---------------------------------------------------
    # 線性回歸類（Regression-based Features）
    # ---------------------------------------------------
    "Slope": lambda x, n: x.rolling(int(n), min_periods=3).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s.dropna()) > 2 else np.nan
    ),  # 回歸斜率
    "Rsquare": lambda x, n: x.rolling(int(n), min_periods=3).apply(
        lambda s: (np.corrcoef(np.arange(len(s)), s)[0, 1] ** 2)
        if s.std() > 0 else np.nan
    ),  # 回歸決定係數 (R²)
    "Resi": lambda x, n: x.rolling(int(n), min_periods=3).apply(
        lambda s: s.iloc[-1] - np.poly1d(np.polyfit(np.arange(len(s)), s, 1))(len(s) - 1)
        if len(s.dropna()) > 2 else np.nan
    ),  # 殘差（實際值 - 線性回歸預測值）

    # ---------------------------------------------------
    # 位置與索引類（Index / Position）
    # ---------------------------------------------------
    "IdxMax": lambda x, n: x.rolling(int(n), min_periods=1).apply(lambda s: np.argmax(s) + 1, raw=True),  # 最大值位置
    "IdxMin": lambda x, n: x.rolling(int(n), min_periods=1).apply(lambda s: np.argmin(s) + 1, raw=True),  # 最小值位置

    # ---------------------------------------------------
    # 雙變數統計類（Pairwise Statistics）
    # ---------------------------------------------------
    "Corr": lambda x, y, n: x.rolling(int(n), min_periods=3).corr(y),  # 相關係數
}


# OPS = {
#     "Greater": np.maximum,
#     "Less": np.minimum,
#     "Abs": np.abs,
#     "Log": np.log,
#     "SignedPower": lambda x, y: np.sign(x) * np.abs(x) ** y,

#     "Mean": lambda x, n: x.rolling(int(n), min_periods=1).mean(),
#     "Std": lambda x, n: x.rolling(int(n), min_periods=1).std(),
#     "Var": lambda x, n: x.rolling(int(n), min_periods=1).var(),
#     "Max": lambda x, n: x.rolling(int(n), min_periods=1).max(),
#     "Min": lambda x, n: x.rolling(int(n), min_periods=1).min(),
#     "Sum": lambda x, n: x.rolling(int(n), min_periods=1).sum(),
#     "Delay": lambda x, n: x.shift(int(n)),
#     "Ref": lambda x, n: x.shift(int(n)),

#     "Rank": lambda x, n=None: (
#         x.rolling(int(n), min_periods=1).apply(lambda s: s.rank(pct=True).iloc[-1])
#         if n else x.rank(pct=True)
#     ),
#     "Quantile": lambda x, n, q: x.rolling(int(n), min_periods=1).quantile(float(q)),

#     "Slope": lambda x, n: x.rolling(int(n), min_periods=3).apply(
#         lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s.dropna()) > 2 else np.nan
#     ),
#     "Rsquare": lambda x, n: x.rolling(int(n), min_periods=3).apply(
#         lambda s: (np.corrcoef(np.arange(len(s)), s)[0, 1] ** 2)
#         if s.std() > 0 else np.nan
#     ),
#     "Resi": lambda x, n: x.rolling(int(n), min_periods=3).apply(
#         lambda s: s.iloc[-1] - np.poly1d(np.polyfit(np.arange(len(s)), s, 1))(len(s) - 1)
#         if len(s.dropna()) > 2 else np.nan
#     ),
#     "Beta": lambda x, y, n: x.rolling(int(n), min_periods=1).cov(y)
#         / (y.rolling(int(n), min_periods=1).var() + 1e-12),

#     "IdxMax": lambda x, n: x.rolling(int(n), min_periods=1).apply(lambda s: np.argmax(s) + 1, raw=True),
#     "IdxMin": lambda x, n: x.rolling(int(n), min_periods=1).apply(lambda s: np.argmin(s) + 1, raw=True),
#     "IMXD": lambda x, n: OPS["IdxMax"](x, n) - OPS["IdxMin"](x, n),

#     "CNTP": lambda x, n: x.diff().fillna(0).gt(0).rolling(int(n), min_periods=1).sum(),
#     "CNTN": lambda x, n: x.diff().fillna(0).lt(0).rolling(int(n), min_periods=1).sum(),
#     "CNTD": lambda x, n: x.diff().fillna(0).eq(0).rolling(int(n), min_periods=1).sum(),

#     "VMA": lambda v, n: OPS["Mean"](v, n) / (v + 1e-12),
#     "VSTD": lambda v, n: OPS["Std"](v, n) / (v + 1e-12),

#     "Cov": lambda x, y, n: x.rolling(int(n), min_periods=3).cov(y),
#     "Corr": lambda x, y, n: x.rolling(int(n), min_periods=3).corr(y),
#     "CORD": lambda x, y, n: (
#         (x / OPS["Ref"](x, 1))
#         .rolling(int(n), min_periods=1)
#         .corr(np.log(y / OPS["Ref"](y, 1)) + 1)
#     ),
# }

# =======================================================
# 🔍 OPS 使用追蹤
# =======================================================
OPS_USED = set()   # 紀錄實際被呼叫過的運算子名稱
OPS_DEFINED = set(OPS.keys())  # 所有已定義的運算子名稱

def report_ops_usage():
    unused = OPS_DEFINED - OPS_USED
    print("\n===============================")
    print("📊 OPS 使用統計")
    print("===============================")
    print(f"✅ 已定義運算子數量：{len(OPS_DEFINED)}")
    print(f"🟢 有被使用：{len(OPS_USED)}")
    print(f"⚪ 未被使用：{len(unused)}")
    print("\n🟢 使用過的運算子：")
    print(", ".join(sorted(OPS_USED)))
    print("\n⚪ 未被使用的運算子：")
    print(", ".join(sorted(unused)))
    print("===============================")

# =======================================================
# 3️⃣ Qlib DSL → Python 轉換
# =======================================================
def qlib_expr_to_python(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"\$(\w+)", r'env["\1"]', expr)
    expr = re.sub(r"\b([A-Z][A-Za-z0-9_]*)\s*\(", r'OPS["\1"](', expr)
    return re.sub(r"\s+", " ", expr)

# =======================================================
# 4️⃣ AST 安全解析器
# =======================================================
class SafeEvaluator(ast.NodeVisitor):
    def __init__(self, env):
        self.env = env

    def visit_Name(self, node):
        if node.id == "env": return self.env
        if node.id == "OPS": return OPS
        if node.id in self.env: return self.env[node.id]
        raise ValueError(f"未知名稱: {node.id}")

    def visit_Constant(self, node): 
        return node.value

    def visit_BinOp(self, node):
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        raise ValueError(f"Unsupported operator: {node.op}")

    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        return -val if isinstance(node.op, ast.USub) else val

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]

        # --- 追蹤是否為 OPS 中的函式 ---
        if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "OPS":
                op_name = self.visit(node.func.slice)
                OPS_USED.add(op_name)  # ✅ 紀錄被使用的運算子
        return func(*args)

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        return getattr(value, node.attr)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        key = self.visit(node.slice)
        
        try:
            return value[key]
        except Exception:
            raise KeyError(f"❌ 字典中找不到鍵：{key}（物件: {type(value).__name__}）")

    # def visit_Compare(self, node):
    #     left = self.visit(node.left)
    #     right = self.visit(node.comparators[0])
    #     op = node.ops[0]
    #     if isinstance(op, ast.Gt):
    #         return (left > right).astype(float)
    #     elif isinstance(op, ast.Lt):
    #         return (left < right).astype(float)
    #     elif isinstance(op, ast.GtE):
    #         return (left >= right).astype(float)
    #     elif isinstance(op, ast.LtE):
    #         return (left <= right).astype(float)
    #     elif isinstance(op, ast.Eq):
    #         return (left == right).astype(float)
    #     elif isinstance(op, ast.NotEq):
    #         return (left != right).astype(float)
    #     else:
    #         raise ValueError(f"Unsupported comparison operator: {op}") 
    #    

# =======================================================
# 5️⃣ 驗證主程式
# =======================================================
def verify_symbol(symbol: str, 
                  mode: str, 
                  year: int = None,
                  date: str = None,
                  base_dir: str = "./data/day1",
                  dataset_dir: str = "./data/qlib_data/day1/generated_datasets"):
    """
    mode: incremental_pretrain / online_ft
    """

    target_files = []

    if mode == "incremental_pretrain":
        # 驗證 train + valid
        target_files = [
            Path(f"{dataset_dir}/incremental_pretrain/train/{year}.csv"),
            Path(f"{dataset_dir}/incremental_pretrain/valid/{year}.csv"),
        ]
    elif mode == "online_ft":
        if not date:
            raise ValueError("❌ online_ft 模式需要 --date 參數")
        target_files = [Path(f"{dataset_dir}/online_ft/{date[:4]}/{date}.csv")]
    else:
        raise ValueError("❌ mode 必須是 incremental_pretrain 或 online_ft")

    raw_path = Path(base_dir) / f"{symbol}.csv"

    for file_path in target_files:
        if not file_path.exists():
            print(f"⚠️ 跳過，檔案不存在：{file_path}")
            continue
        print(f"\n=== 🧩 驗證檔案：{file_path} ===")
        verify_single_file(symbol, raw_path, file_path)

# =======================================================
# 6️⃣ 單一檔案驗證函數
# =======================================================
def verify_single_file(symbol: str, raw_path: Path, result_path: Path):
    if not raw_path.exists():
        print(f"❌ 找不到原始資料檔案：{raw_path}")
        return

    # --- 讀入原始股票資料 ---
    raw = pd.read_csv(raw_path)
    raw["datetime"] = pd.to_datetime(raw["k_datetime"])
    raw.set_index("datetime", inplace=True)
    raw = raw[~raw.index.duplicated(keep="first")]
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw["vwap"] = (raw["high"] + raw["low"] + raw["close"]) / 3
    env = {c: raw[c] for c in ["open", "high", "low", "close", "volume", "vwap"]}

    # --- 載入 Qlib 結果 ---
    qlib_result = pd.read_csv(result_path)
    qlib_result.columns = [c.strip() for c in qlib_result.columns]
    qlib_result["symbol"] = qlib_result["symbol"].astype(str).str.strip()
    qlib_result["date"] = pd.to_datetime(qlib_result["date"])
    qlib_result.set_index("date", inplace=True)
    qlib_result = qlib_result[qlib_result["symbol"] == str(symbol).strip()].copy()

    if qlib_result.empty:
        print(f"⚠️ 無 symbol={symbol} 的資料。")
        return

    handler = Alpha158(instruments=[symbol], start_time="1980-01-01", end_time="2025-12-31")
    feature_expressions, feature_names = handler.get_feature_config()

    evaluator = SafeEvaluator(env)
    out, compare_all = [], []
    for name, expr in zip(feature_names, feature_expressions):
        if expr is None:
            print(f"\n=== Debug {name} ===\n⚠️ 找不到公式")
            continue

        print(f"\n=== Debug {name} ===")
        print("原始公式:", expr)

        try:
            py_expr = qlib_expr_to_python(expr)
            print("轉換後公式:", py_expr)

            tree = ast.parse(py_expr, mode="eval")
            ours = evaluator.visit(tree.body)
        except Exception as e:
            ours = None
            print(f"{name} ❌ 錯誤:", e)

        if ours is None or name not in qlib_result.columns:
            out.append((name, None))
            continue

        s1, s2 = ours.align(qlib_result[name], join="inner")
        if len(s1) == 0:
            out.append((name, None))
            continue

        mae = np.nanmean(np.abs(s1 - s2))
        out.append((name, mae))
        compare_all.append(pd.DataFrame({
            "feature": name, 
            "date": s1.index,
            "our_calc": s1.values,
            "qlib_calc": s2.values,
            "diff": (s1 - s2).values
        }))

    out_df = pd.DataFrame(out, columns=["feature", "MAE"])
    out_path = Path(f"./verify_{result_path.parent.stem}_result_{symbol}_{result_path.stem}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ 指標誤差統計輸出：{out_path}")

    if compare_all:
        compare_out_path = Path(f"./verify_{result_path.parent.stem}_result_compare_{symbol}_{result_path.stem}.csv")
        pd.concat(compare_all, ignore_index=True).to_csv(compare_out_path, index=False)
        print(f"✅ 完整對照輸出：{compare_out_path}")

# =======================================================
# 7️⃣ 執行入口
# =======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--mode", choices=["incremental_pretrain", "online_ft"], required=True)
    parser.add_argument("--year", type=int, help="incremental_pretrain 年份")
    parser.add_argument("--date", type=str, help="online_ft 日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    verify_symbol(args.symbol, mode=args.mode, year=args.year, date=args.date)
    report_ops_usage()
