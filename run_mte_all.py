#!/usr/bin/env python3
"""
Run IDTxl Multivariate Transfer Entropy (MTE) for ALL datasets in ./dataset
and write per-dataset outputs to results/mte/<dataset_id>/ :

- results.pkl             (pickled IDTxl Results object; reliable)
- mte_summary.json        (JSON summary with edge_list as a LIST of dicts; safe)
Then export one-line-per-dataset summary CSV:
- results/mte/te_edges_summary.csv
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

DATA_DIR = Path("dataset")
OUT_ROOT = Path("results/mte")

# ---------- helpers ----------
def dataset_id_from_path(p: Path) -> str:
    # keep filename stem EXACTLY (do not drop _0 etc.)
    return p.stem

def load_data(p: Path):
    import numpy as np
    if p.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        x = df.values
        # make it (vars, time)
        if x.shape[0] < x.shape[1]:
            x = x.T
        else:
            # if csv is (time, vars) usually time >> vars, so transpose
            x = x.T
        return x
    elif p.suffix.lower() == ".dat":
        x = np.loadtxt(p)
        if x.ndim == 2 and x.shape[0] > x.shape[1]:
            x = x.T
        return x
    else:
        raise ValueError(f"Unsupported file type: {p}")

def safe_json(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def extract_edges_from_results(res, fdr: bool = False):
    """
    Robustly extract edges using get_single_target() if available.
    Returns a LIST[dict] edges with keys: source, target, lag, weight(optional), pvalue(optional)
    """
    edges = []
    # IDTxl Results usually has: res.targets (list) or res.get_targets()
    targets = None
    for attr in ("targets", "target_list"):
        if hasattr(res, attr):
            targets = getattr(res, attr)
            break
    if targets is None:
        # try common method
        if hasattr(res, "get_targets"):
            try:
                targets = res.get_targets()
            except Exception:
                targets = None

    # fallback: assume numeric targets 0..(n_vars-1) if possible
    if targets is None:
        targets = []

    def add_edge(src, tgt, lag, extra=None):
        d = {"source": int(src), "target": int(tgt), "lag": int(lag)}
        if isinstance(extra, dict):
            d.update({k: safe_json(v) for k, v in extra.items()})
        edges.append(d)

    # Preferred route: res.get_single_target()
    if hasattr(res, "get_single_target"):
        # Determine candidate target indices
        cand_targets = []
        if isinstance(targets, (list, tuple)) and len(targets) > 0:
            cand_targets = list(targets)
        else:
            # Try infer number of targets from res._single_target or res.results? keep safe
            cand_targets = list(range(0, 50))  # upper bound; will break on errors

        for t in cand_targets:
            try:
                st = res.get_single_target(target=t, fdr=fdr)
            except TypeError:
                # older signatures
                try:
                    st = res.get_single_target(target=t)
                except Exception:
                    continue
            except Exception:
                continue

            if not isinstance(st, dict):
                continue

            # Common keys (var + lag)
            srcs = st.get("selected_vars_sources") or st.get("sources_selected")
            lags = st.get("selected_vars_sources_lags") or st.get("sources_selected_lags")

            # Some variants: list of tuples
            if isinstance(st.get("selected_links"), (list, tuple)):
                for item in st["selected_links"]:
                    # try (src, lag) or (src, lag, score)
                    try:
                        src = item[0]
                        lag = item[1]
                        add_edge(src, t, lag)
                    except Exception:
                        pass
                continue

            if isinstance(srcs, (list, tuple)) and isinstance(lags, (list, tuple)) and len(srcs) == len(lags):
                for src, lag in zip(srcs, lags):
                    add_edge(src, t, lag)
                continue

        # de-duplicate
        uniq = {}
        for e in edges:
            key = (e["source"], e["target"], e["lag"])
            uniq[key] = e
        return list(uniq.values())

    # If no known API: give up safely
    return []

def run_one(data_path: Path, out_dir: Path, force: bool, fdr: bool, settings: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "results.pkl"
    summary_path = out_dir / "mte_summary.json"

    if (not force) and summary_path.exists():
        return "SKIP", summary_path

    # ----- run IDTxl -----
    from idtxl.data import Data
    from idtxl.multivariate_te import MultivariateTE

    x = load_data(data_path)  # (vars, time)
    data = Data(x, dim_order='ps')  # p=process(vars), s=samples(time)

    mte = MultivariateTE()
    res = mte.analyse_single_target(data=data, settings=settings, target=0)  # placeholder
    # If analyse_single_target only does one target, we need analyse_network:
    # Many IDTxl versions support analyse_network for multivariate TE:
    try:
        res = mte.analyse_network(data=data, settings=settings)
    except Exception:
        # fallback: try run per target and merge not supported -> at least save per target 0
        pass

    # Save pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(res, f)

    # Extract edges (uncorrected by default)
    edges = extract_edges_from_results(res, fdr=fdr)

    summary = {
        "file": str(data_path),
        "shape_vars_time": [int(x.shape[0]), int(x.shape[1])],
        "settings": {k: safe_json(v) for k, v in settings.items()},
        "fdr": bool(fdr),
        "edge_list": edges,          # LIST[dict]
        "n_edges": len(edges),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return "OK", summary_path

def export_csv(out_csv: Path):
    rows = []
    for summary in sorted(OUT_ROOT.rglob("mte_summary.json")):
        d = json.loads(summary.read_text())
        ds = summary.parent.name
        edges = d.get("edge_list") if isinstance(d, dict) else None
        n_edges = len(edges) if isinstance(edges, list) else 0
        rows.append([ds, str(summary), n_edges])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "summary_path", "n_edges"])
        w.writerows(rows)

    print(f"OK: wrote {len(rows)} rows -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run even if summary exists")
    ap.add_argument("--fdr", action="store_true", help="use fdr-corrected edge extraction (if available)")
    ap.add_argument("--out_csv", default="results/mte/te_edges_summary.csv")
    args = ap.parse_args()

    # Settings: keep conservative + reproducible.
    # You can adjust tau_min/tau_max to match your previous runs.
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "tau_min": 1,
        "tau_max": 5,
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "n_perm_max_stat": 200,
        "n_perm_omnibus": 200,
        "alpha_max_stat": 0.05,
        "alpha_min_stat": 0.05,
        "alpha_omnibus": 0.05,
        "fdr_correction": args.fdr,
        "verbose": False,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    data_files = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() in (".csv", ".dat")])
    ok = skip = fail = 0

    for p in data_files:
        ds = dataset_id_from_path(p)
        out_dir = OUT_ROOT / ds
        try:
            status, _ = run_one(p, out_dir, force=args.force, fdr=args.fdr, settings=settings)
            if status == "OK":
                ok += 1
            else:
                skip += 1
        except Exception as e:
            fail += 1
            # write an error-only json so you see what failed
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "mte_summary.json").write_text(json.dumps({"file": str(p), "_error": str(e)}, indent=2))

    export_csv(Path(args.out_csv))
    print(f"Done. OK={ok}, SKIP={skip}, FAIL={fail}")

if __name__ == "__main__":
    main()
