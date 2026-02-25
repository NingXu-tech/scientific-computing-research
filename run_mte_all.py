#!/usr/bin/env python3
"""
Run IDTxl Multivariate TE for all datasets and write:
- results/mte/<dataset_name>/mte_summary.json   (contains edge list)
- results/mte/te_edges_summary.csv             (one row per dataset)
"""
import argparse, json, csv, os
from pathlib import Path

def infer_dataset_name(p: Path) -> str:
    return p.stem.replace("_test","")

def load_data_file(p: Path):
    # supports csv and dat (simple whitespace)
    import numpy as np
    if p.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        # assume columns are variables
        x = df.values.T  # (vars, time)
        return x
    elif p.suffix.lower() == ".dat":
        import numpy as np
        x = np.loadtxt(p)
        # if shape is (time, vars) -> transpose
        if x.ndim == 2 and x.shape[0] > x.shape[1]:
            x = x.T
        return x
    else:
        raise ValueError(f"Unsupported data file: {p}")

def run_mte(x, settings):
    from idtxl.multivariate_te import MultivariateTE
    mte = MultivariateTE()
    # IDTxl expects data as numpy array (vars, time)
    results = mte.analyse_network(settings=settings, data=x)
    return results

def edges_from_results(results, fdr: bool):
    # get adjacency matrix from results, then edge list
    adj = results.get_adjacency_matrix(weights="max_te_lag", fdr=fdr)
    # IDTxl adjacency exposes get_edge_list(); do NOT use .matrix
    edge_list = []
    for src, dst, w in adj.get_edge_list():
        edge_list.append({"src": int(src), "dst": int(dst), "weight": float(w)})
    return edge_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset", help="dataset folder")
    ap.add_argument("--out_root", default="results/mte", help="output root folder")
    ap.add_argument("--out_csv", default="results/mte/te_edges_summary.csv", help="summary csv")
    ap.add_argument("--force", action="store_true", help="re-run even if mte_summary.json exists")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # settings: adjust if your group has fixed params
    settings = dict(
        cmi_estimator="JidtGaussianCMI",
        max_lag_sources=5,
        min_lag_sources=1,
        n_perm_max_stat=200,
        n_perm_omnibus=200,
        alpha_max_stat=0.05,
        alpha_min_stat=0.05,
        alpha_omnibus=0.05,
        fdr_correction=False,
        tau_min=1,
        tau_max=5,
        verbosity=0,
        verbose=False,
        write_ckp=False,
    )

    # datasets list: analytic + brownian y1~y3 + sim-ex1~2 + synthetic 01~05 v1~v4 = 26
    wanted = []
    wanted.append(data_dir / "analytic-ex1.dat")
    wanted += [data_dir / f"brownian-data-y{i}.dat" for i in [1,2,3]]
    wanted += [data_dir / "sim-ex1.dat", data_dir / "sim-ex2.dat"]
    for eq in range(1,6):
        for v in range(1,5):
            wanted.append(data_dir / f"synthetic-data-0{eq}-v{v}.csv")

    rows = []
    ok = skip = fail = 0

    for fp in wanted:
        name = infer_dataset_name(fp)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "mte_summary.json"

        if summary_path.exists() and (not args.force):
            # still include in csv
            d = json.loads(summary_path.read_text())
            # edge list key preference
            edges = d.get("edge_list_max_te_lag_fdr_false")
            n_edges = len(edges) if isinstance(edges, list) else 0
            rows.append([name, str(summary_path), "edge_list_max_te_lag_fdr_false", n_edges, "SKIP(existing)"])
            skip += 1
            continue

        try:
            x = load_data_file(fp)
            results = run_mte(x, settings)

            edge_false = edges_from_results(results, fdr=False)
            # fdr=True may fail if no edges; handle safely
            try:
                edge_true = edges_from_results(results, fdr=True)
            except Exception as e:
                edge_true = {"_error": str(e)}

            summary = dict(
                file=str(fp),
                shape_vars_time=[int(x.shape[0]), int(x.shape[1])],
                settings=settings,
                edge_list_max_te_lag_fdr_false=edge_false,
                edge_list_max_te_lag_fdr_true=edge_true,
            )
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

            rows.append([name, str(summary_path), "edge_list_max_te_lag_fdr_false", len(edge_false), "OK"])
            ok += 1
        except Exception as e:
            # write error summary so you can debug per dataset
            summary = dict(
                file=str(fp),
                settings=settings,
                edge_list_max_te_lag_fdr_false={"_error": str(e)},
                edge_list_max_te_lag_fdr_true={"_error": str(e)},
            )
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            rows.append([name, str(summary_path), "", 0, f"FAIL: {e}"])
            fail += 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","summary_path","edge_list_key","n_edges","note"])
        w.writerows(rows)

    print(f"Done. OK={ok}, SKIP={skip}, FAIL={fail}")
    print(f"Wrote: {out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
