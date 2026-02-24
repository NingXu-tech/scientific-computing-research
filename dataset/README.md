# HMR3 GPCR Unbinding Trajectories

```python
with open("hmr3_traj_important_features.npy","rb") as f:
    traj = np.load(f)
    labels = np.load(f)
```

traj -> np.array with shape of 148 x 25000 x 6
- 148 -- simulation replicas
- 25000 -- simulation frames - ligand - residue distances
- 6 -- individual residues (L482, T234, F224, S226, Wat565, Wat569, in order)

labels -> 1D np.array with shape of 148
- 148 -- simulation outcomes (IN vs OUT)