# README.md

## Project Description

This repository contains the code and experimental results for causal discovery methods applied to multiple datasets. The implemented methods include **PC, GES, Granger, VARLiNGAM, Transfer Entropy, CCM, and PCMCI**.

---

## File Structure

```
FINAL DATA/
│
├── dataset/                  
│   ├── analytic-ex1.dat               
│   ├── brownian-data-y{1,2,3}_0.dat   
│   ├── hmr3_traj_important_features.npy  
│   ├── README.md 
│   ├── sim-ex1.dat, sim-ex2.dat        
│   └──synthetic-data-XX-vY.csv       
│   
│
├── analytic-ex1_test.ipynb                  
├── brownian-data-y_test.ipynb               
├── hmr3_traj_important_features_test.ipynb 
├── sim-ex_test.ipynb                        
└── synthetic-data_test.ipynb    
└── requirements.txt       
└── README.md                      
```

---

## Dependencies

This project is based on **Python 3.10+**. Key dependencies can be installed via `pip install -r requirements.txt`:

* `numpy`
* `pandas`
* `matplotlib`
* `networkx`
* `statsmodels`
* `tigramite` (PCMCI method)
* `idtxl` (Transfer Entropy method)
* `causal-learn` (PC, GES, LiNGAM methods)
* `pyEDM` (Convergent Cross Mapping)
* `tqdm` (progress bar)
* `jupyter`

```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Wangxinping-gif/PHAS0077-Scientific-Computing-Individual-Research-Project-24-25.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open Jupyter Notebook:

```bash
jupyter notebook
```

4. Run the appropriate notebook for your experiment:

* `analytic-ex1_test.ipynb` — Analytical example dataset
* `brownian-data-y_test.ipynb` — Brownian motion datasets
* `synthetic-data_test.ipynb` — Synthetic datasets
* `hmr3_traj_important_features_test.ipynb` — GPCR data

---

## Dataset Description

* **analytic-ex1.dat / sim-ex1.dat / sim-ex2.dat**
  Simulation datasets used to verify the correctness of causal discovery methods.

* **brownian-data-y{1,2,3}\_0.dat**
  Brownian motion datasets with delayed dependencies, used to test time-series causal discovery under noise.


* **synthetic-data-XX-vY.csv**
  Artificial multivariate time-series datasets (multiple versions) for systematic comparison of causal discovery methods.

* **hmr3\_traj\_important\_features.npy**
  Real GPCR trajectory feature data used for actual causal structure inference.
  
---

## References and Acknowledgements

* Some implementations are based on the following libraries:

  * **causal-learn**: [https://github.com/cmu-phil/causal-learn](https://github.com/cmu-phil/causal-learn)
  * **tigramite**: [https://github.com/jakobrunge/tigramite](https://github.com/jakobrunge/tigramite)
  * **IDTxl**: [https://github.com/pwollstadt/IDTxl](https://github.com/pwollstadt/IDTxl)