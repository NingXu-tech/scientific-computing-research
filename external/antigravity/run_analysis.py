
import numpy as np
import csv
from generators import DatasetA, DatasetB, DatasetC, DatasetD, DatasetE
from statistics import DoStatistics

def run_dataset_analysis(name, GeneratorClass, length=10000, seeds=range(10), discretization_bins=10):
    print(f"\n--- Analyzing {name} ---")
    
    results = []
    stats_engine = DoStatistics(discretization_bins=discretization_bins)
    
    for seed in seeds:
        print(f"Seed {seed}...")
        gen = GeneratorClass(length=length, seed=seed)
        baseline_data = gen.generate()
        T, d = baseline_data.shape
        
        for target_idx in range(d):
            # Clamp to 2.0
            mu = 0.0
            kappa = 0.2
            sigma = 0.2
            rng = np.random.default_rng(seed + 1000*target_idx)

            def policy(t, hist):
                # hist may be a vector (d,), a matrix (t,d), or a python list.
                # We always extract the CURRENT value of the intervened variable.
                if hist is None:
                    x = mu
                else:
                    try:
                        # numpy array case
                        import numpy as _np
                        h = _np.asarray(hist)
                        if h.ndim == 0:
                            x = float(h)
                        elif h.ndim == 1:
                            x = float(h[target_idx])
                        else:
                            x = float(h[-1, target_idx])
                    except Exception:
                        # list/other case
                        x = float(hist[-1][target_idx]) if len(hist) else mu

                val = x + kappa*(mu - x) + float(rng.normal(0.0, sigma))
                return float(val)
            gen_int = GeneratorClass(length=length, seed=seed)
            gen_int.set_intervention(target_idx, policy)
            intervened_data = gen_int.generate()
            
            # Save Datasets to CSV
            # Baseline (only need to save once per seed, but overwriting is fine or check existence)
            if target_idx == 0:
                base_fname = f"generated_data/{name}_Seed{seed}_Baseline.csv"
                np.savetxt(base_fname, baseline_data, delimiter=",", header="X,Y,Z", comments="")
            
            int_fname = f"generated_data/{name}_Seed{seed}_InterventionOnVar{target_idx}.csv"
            np.savetxt(int_fname, intervened_data, delimiter=",", header="X,Y,Z", comments="")

            lags = [1, 10, 50]
            for lag_val in lags:
                djs = stats_engine.compute_markov_divergence(baseline_data, intervened_data, lag=lag_val)
                # Spectral stats
                dstat, dkin = stats_engine.compute_spectral_divergence(baseline_data, intervened_data, lag=lag_val)
                
                res = {
                    "dataset": name,
                    "seed": seed,
                    "intervention_target": target_idx,
                    "lag": lag_val,
                    "DeltaJS": djs,
                    "DeltaStat": dstat,
                    "DeltaKin": dkin,
                }
                
                for j in range(d):
                    # Use sufficient lag to cover the dependencies in Dataset C/D (max lag 15)
                    # For Mechanism Shift, we keep max_lag fixed at 15 for now as it's a VAR model property
                    if lag_val == 1: # Only compute expensive DLL once per seed/target, or just duplicate?
                        # Let's compute it once and associate with lag=1 row, or just include it always.
                        # Actually DLL depends on 'max_lag' parameter of the probe, not the transition lag.
                        # We'll just run it.
                        dll = stats_engine.compute_mechanism_shift_var(baseline_data, intervened_data, target_idx=j, max_lag=15)
                        res[f"DLL_var{j}"] = dll
                    else:
                        res[f"DLL_var{j}"] = "" # Empty for other lags to save time/space
                    
                    # Compute Change in Transfer Entropy (Delta TE)
                    # TE also has a lag parameter. We should use lag_val.
                    if j != target_idx:
                         te_base = stats_engine.compute_transfer_entropy(baseline_data, source_idx=target_idx, target_idx=j, lag=lag_val)
                         te_int = stats_engine.compute_transfer_entropy(intervened_data, source_idx=target_idx, target_idx=j, lag=lag_val)
                         res[f"DeltaTE_XtoVar{j}"] = te_base - te_int
                    else:
                         res[f"DeltaTE_XtoVar{j}"] = 0.0
                    
                results.append(res)
            
    return results

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Run only specific dataset")
    args = parser.parse_args()

    datasets_all = {
        "DatasetA": DatasetA,
        "DatasetB": DatasetB,
        "DatasetC": DatasetC,
        "DatasetD": DatasetD,
        "DatasetE": DatasetE
    }
    
    if args.dataset:
        if args.dataset not in datasets_all:
            raise ValueError(f"Unknown dataset {args.dataset}")
        datasets = {args.dataset: datasets_all[args.dataset]}
        out_name = f"do_statistics_results_{args.dataset}.csv"
    else:
        datasets = datasets_all
        out_name = "do_statistics_results.csv"
    
    all_results = []
    for name, cls in datasets.items():
        res = run_dataset_analysis(name, cls, length=5000)
        all_results.extend(res)
        
    # Save to CSV
    # Collect all unique keys across all results (since different datasets have different vars)
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    
    # Sort keys for consistent columns (Dataset, Seed, Intervention usually first)
    # Custom sort: Metadata first, then metrics
    meta_keys = ['dataset', 'seed', 'intervention_target']
    metric_keys = sorted([k for k in all_keys if k not in meta_keys])
    keys = meta_keys + metric_keys

    with open(out_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\nDetailed results saved to {out_name}")
    
    # Simple summary printing
    # Group by dataset, intervention_target
    from collections import defaultdict
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    
    for r in all_results:
        key = (r['dataset'], r['intervention_target'])
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ['seed', 'intervention_target']:
                sums[key][k] += v
        counts[key]['count'] += 1
        
    print("\n=== Summary Results (Averaged over seeds) ===")
    print(f"{'Dataset':<10} {'IntTgt':<8} {'DeltaJS':<10} {'DLL_v0':<10} {'DLL_v1':<10} {'DLL_v2':<10}")
    for (dset, tgt), val_dict in sorted(sums.items()):
        n = counts[(dset, tgt)]['count']
        djs = val_dict['DeltaJS'] / n
        dll0 = val_dict['DLL_var0'] / n
        dll1 = val_dict['DLL_var1'] / n
        dll2 = val_dict['DLL_var2'] / n
        print(f"{dset:<10} {tgt:<8} {djs:<10.4f} {dll0:<10.4f} {dll1:<10.4f} {dll2:<10.4f}")
