import pandas as pd
from pathlib import Path
from collections import Counter

path = Path(__file__).resolve().parents[1] / "DATA" / "recuima-020425.xlsx"
df = pd.read_excel(path)
col = "complicaciones"
if col not in df.columns:
    print("Column 'complicaciones' not found")
    raise SystemExit

vals = df[col].dropna().astype(str)
norm = vals.str.lower().str.strip()
split_vals = norm.str.split(r"[;|/,]+")
ctr = Counter()
for parts in split_vals:
    for t in parts:
        t = t.strip()
        if t:
            ctr[t] += 1

print("Total non-null rows:", len(vals))
print("Unique tokens:", len(ctr))
print("Top 60 tokens:")
for tok, c in ctr.most_common(60):
    print(f"{tok}: {c}")
