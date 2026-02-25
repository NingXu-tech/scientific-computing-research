#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def flatten_edges(d):
    # Try common keys used by various MTE/IDTxl summaries.
    candidate_keys = [
        "edge_list_max_te_lag_fdr_false",
        "edge_list_max_te_lag_fdr_true",
        "edge_list_fdr_false",
        "edge_list_fdr_true",
        "edge_list",
        "edges",
    ]
    for k in candidate_keys:
        if isinstance(d, dict) and k in d and isinstance(d[k], (list, tuple)):
            return k, list(d[k])
    return None, []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mte", help="root folder containing */mte_summary.json")
    ap.add_argument("--out_csv", default="results/mte/te_edges_summary.csv", help="output CSV path")
    args = ap.parse_args()

    root = Path(args.root)
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for js in sorted(root.rglob("mte_summary.json")):
        d = load_json(js)
        if not isinstance(d, dict):
            continue
        key_used, edges = flatten_edges(d)

        dataset = js.parent.name  # e.g., synthetic-data-03-v1
        # Capture settings if available
        settings = d.get("settings", {}) if isinstance(d.get("settings", {}), dict) else {}

        # If no edge list found, still write a row (useful for auditing)
        if not edges:
            rows.append({
                "dataset": dataset,
                "summary_path": str(js),
                "edge_list_key": key_used or "",
                "n_edges": 0,
                "note": "no edge list key found",
            })
            continue

        for e in edges:
            r = {
                "dataset": dataset,
                "summary_path": str(js),
                "edge_list_key": key_used or "",
            }
            if isinstance(e, dict):
                # Keep common fields if present
                for kk in ["source", "src", "from", "target", "tgt", "to", "lag", "delay", "te", "value", "score", "p_value", "pval", "alpha"]:
                    if kk in e:
                        r[kk] = e[kk]
                # If dict has other keys, store them as JSON string for traceability
                r["edge_raw"] = json.dumps(e, ensure_ascii=False)
            else:
                # edge might be tuple/list like (src, tgt, lag, te, p)
                r["edge_raw"] = json.dumps(e, ensure_ascii=False)
            # store a few settings (optional)
            if settings:
                for kk in ["alpha", "max_lag", "min_lag", "fdr", "estimator"]:
                    if kk in settings:
                        r[f"settings_{kk}"] = settings[kk]
            rows.append(r)

    # Decide header
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"OK: wrote {len(rows)} rows -> {outp}")

if __name__ == "__main__":
    main()
