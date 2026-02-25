#!/usr/bin/env python3
import csv, pickle, re, io
from pathlib import Path
from contextlib import redirect_stdout

ROOT = Path("results/mte")
OUT  = ROOT / "te_edges_long.csv"

EDGE_RE = re.compile(r'^\s*(\d+)\s*->\s*(\d+),\s*([A-Za-z0-9_]+):\s*([\-0-9\.eE]+)\s*$')

def parse_edge_text(txt: str):
    edges=[]
    for line in txt.splitlines():
        m = EDGE_RE.match(line.strip())
        if not m:
            continue
        edges.append({
            "src": int(m.group(1)),
            "dst": int(m.group(2)),
            "key": m.group(3),
            "val": float(m.group(4)),
        })
    return edges

def print_edge_list(res, weights: str, fdr: bool):
    buf = io.StringIO()
    with redirect_stdout(buf):
        res.print_edge_list(weights=weights, fdr=fdr)
    return buf.getvalue()

def main():
    rows=[]
    # 你已经验证过：max_te_lag 可用；其它权重我们尽力尝试（有就写，没有就空）
    weight_candidates = ["max_te_lag", "max_te", "te", "p_value", "pval", "statistic"]

    for dsdir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        ds = dsdir.name
        pkl = dsdir / "results.pkl"
        if not pkl.exists():
            continue

        res = pickle.load(open(pkl, "rb"))

        # 先拿 lag（一定要）
        lag_txt = print_edge_list(res, weights="max_te_lag", fdr=False)
        lag_edges = parse_edge_text(lag_txt)

        # 其它权重（可选）
        extra = {}
        for w in weight_candidates:
            if w == "max_te_lag":
                continue
            try:
                txt = print_edge_list(res, weights=w, fdr=False)
                eds = parse_edge_text(txt)
                # 用 (src,dst) 做 key
                extra[w] = {(e["src"], e["dst"]): e["val"] for e in eds}
            except Exception:
                extra[w] = {}

        # 写行：每条边一行
        for e in lag_edges:
            src, dst = e["src"], e["dst"]
            lag = e["val"]
            row = {
                "dataset": ds,
                "src": src,
                "dst": dst,
                "lag": lag,
                "max_te": extra.get("max_te", {}).get((src,dst), ""),
                "te":     extra.get("te", {}).get((src,dst), ""),
                "p_value": extra.get("p_value", {}).get((src,dst), extra.get("pval", {}).get((src,dst), "")),
                "statistic": extra.get("statistic", {}).get((src,dst), ""),
            }
            rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset","src","dst","lag","max_te","te","p_value","statistic"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"OK: wrote {len(rows)} edge-rows -> {OUT}")

if __name__ == "__main__":
    main()
