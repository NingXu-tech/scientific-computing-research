import os
import numpy as np
import pandas as pd

from te_opt.te_runner import TERunConfig, run_multivariate_te

cfg = TERunConfig(verbosity=0)

cases = [
    ("brownian-y2", lambda: pd.read_csv("./dataset/brownian-data-y2_0.dat",
                                        delim_whitespace=True, header=None, usecols=[1,2]).iloc[:400].values.T),
    ("analytic-ex1", lambda: pd.read_csv("./dataset/analytic-ex1.dat",
                                         delim_whitespace=True).iloc[:400].values.T),
    ("synthetic-01-v1", lambda: pd.read_csv("./dataset/synthetic-data-01-v1.csv").iloc[:2000].values.T),
]

out_dir="te_opt/out_regression_final"
os.makedirs(out_dir, exist_ok=True)

all_ok = True
for name, loader in cases:
    data_array = loader()
    res = run_multivariate_te(data_array, cfg=cfg)

    # 1) native binary adjacency (来自 idtxl)
    native = np.asarray(res.get_adjacency_matrix(weights="binary", fdr=False)).astype(int)

    # 2) “模拟你的导出”：只导出 binary
    out_fp = f"{out_dir}/{name}_binary_fdr0.csv"
    np.savetxt(out_fp, native, fmt="%d", delimiter=",")

    exported = np.loadtxt(out_fp, delimiter=",").astype(int)

    ok = np.array_equal(native, exported)
    print(f"{name}: equal? {ok}")
    all_ok = all_ok and ok

print("ALL PASS" if all_ok else "SOME FAIL")
raise SystemExit(0 if all_ok else 1)
