## OATS-MCS: SCOPF Monte Carlo Sampling Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18792808.svg)](https://doi.org/10.5281/zenodo.18792808) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Version:** v1.0.0
**Release Date:** February 2026
**Status:** Stable (validated release)
**License:** GPLv3

Monte Carlo sampling extension for the OATS framework, enabling scalable Preventive DC Security-Constrained Optimal Power Flow (SCOPF) scenario generation with renewable energy (RES) integration. The design supports deterministic block-parallel expansion to large sample sizes and flexible adaptation to different network scales.

This implementation reflects the core dataset generation methodology described in the associated dataset paper.

---

## 1. Project Structure

```
oats_sc/                # Preventive SCOPF model (OATS-derived)
oats_mcs/               # Scenario sampling framework
oats_val/               # Dataset validation utilities
oats_ml/                # Neural network demo training
IEEE-24R/               # Base testcase
IEEE24_demo_30r/        # Minimal demo dataset
OATS_MCS_result/        # Sampling outputs
env/                    # Conda environment files
```

---

## 2. Core Methodology

### 2.1 Core Capabilities

* Preventive DC-SCOPF formulation
* Renewable (RES) injection modeling
* Curtailment penalty support
* Sequential scenario sampling

---

### 2.2 Supported Sampling Modes

Sampling mode is specified via string flag:

```
uniform
sobol
multi_uniform
multi_lhs
lhs_<N>
```

Where:

* `uniform` → Standard Monte Carlo sampling
* `sobol` → Quasi–Monte Carlo (Sobol sequence)
* `multi_uniform` → Multi-block uniform sampling
* `multi_lhs` → Multi-block Latin Hypercube sampling
* `lhs_<N>` → Latin Hypercube sampling with total sample size N
* Example: `lhs_10000`

For `uniform`, `lhs_<N>`, and `sobol`, block-parallel generation uses structured seed encoding:

```
rand_seed = seed_main * 1000 + block_id
# last 3 digits = block index
```

* `seed_main` controls the global distribution.
* `block_id` identifies the sampling block.
* This design ensures deterministic, reproducible multi-block generation without altering statistical properties.
* For renewable (RES) sampling, an additional seed shift is applied to decorrelate it from demand sampling:

  ```
  rand_seed_res = rand_seed + rand_N
  ```

---

### 2.3 Dataset Structure

Generated datasets follow a unified relational schema (SQLite tables):

```
Generator_Real
Demand_Real
PowerFlow
RES_Real
RES_UB
RES_LB
```

Primary storage format:

* SQLite

Optional exports:

* HDF5; Parquet; CSV; XLSX

---

## 3. Test Case (IEEE-24R)

The IEEE-24R test case is derived from the IEEE 24-bus RTS benchmark and incorporates updated network and renewable capacity data based on the DTU 2016 revision.

### 3.1 Model Adjustments

* Preventive DC-SCOPF formulation in Pyomo model.
* Small perturbations are applied to generator cost coefficients to mitigate cost degeneracy and enforce unique dispatch solutions.
* N-1 contingency set is predefined and screened for feasibility to ensure numerical stability and consistent validation.

### 3.2 Included Files

* `IEEE-24R/IEEE-24R.xlsx` — network, generator, load, and renewable data
* `IEEE-24R/N-1_contingency_test/` — contingency configuration parameters
* `IEEE-24R/IEEE-24R.svg` — single-line diagram (vector format)
* `IEEE-24R/IEEE-24R.pdf` — single-line diagram 
* `IEEE-24R/results.xlsx` — single-scenario output example

---

## 4. Validation

### 4.1 Validation Reports

For each validated SQLite dataset, a corresponding summary report is generated:

```
validation_reports/<dataset_name>.xlsx
```

The report includes:

* Feasibility statistics
* Generator (PG) bound violation summary
* Renewable (PR/PW) bound violation summary
* Power balance residual statistics
* PTDF-based flow reconstruction error metrics

---

## 5. Environment

### 5.1 Sampling (OATS-MCS)

Required:

* Python 3.11.10
* Pyomo 6.9.2
* Gurobi 13.0 (licensed)

```
conda env create -f oatsmcs.yml
conda activate oatsmcs
```

---

### 5.2 Validation / NN Demo (separate environment)

Validation and NN training are tested under a separate environment:

```
conda env create -f oatsmcs_nn.yml
conda activate oatsmcs_nn
```

---

## 6. Execution

### 6.1 Scenario Sampling

```
python run_oatsmcs.py
```

Generates sampled SCOPF scenarios and exports structured dataset files.

---

### 6.2 Single-Scenario Analysis

```
python run_oats_sc.py
```

Runs a single preventive SCOPF instance and produces:

* `result.xlsx` — single scenario output.

---

### 6.3 Dataset Validation

```
python run_validation.py
```

Performs numerical, structural, and dispatch-level consistency checks on generated datasets.

---

### 6.4 Neural Network Demo

```
python run_nn_demo.py
```

Runs baseline neural network training on the demo dataset.

---

## 7. Author & Citation

**Runsheng He** Department of Electronic & Electrical Engineering, University of Strathclyde

Released: February 2026

### Dataset & Software Attribution

This project consists of three core original contributions. If you use any of these components, please acknowledge the respective work:

- **IEEE-24R System**: A refined power system benchmark developed and curated by Runsheng He, adapted from the IEEE 24-bus Reliability Test System (RTS) benchmark, implemented via [MATPOWER](https://ieeexplore.ieee.org/document/5491276) and incorporating updates from the [DTU 2016 revised RTS version](https://backend.orbit.dtu.dk/ws/portalfiles/portal/120568114/An_Updated_Version_of_the_IEEE_RTS_24Bus_System_for_Electricty_Market_an....pdf).
- **OATS-MCS (including Validation)**: A Monte Carlo sampling and physical-consistency validation framework developed by Runsheng He for large-scale, feasible scenario generation.
- **OATS-ML**: A neural network workflow created by Runsheng He, specifically designed for proxy modeling and performance benchmarking of the SCOPF dataset.

### How to Cite

If you use the **IEEE-24R** dataset, the **OATS-MCS** sampling/validation logic, or the **OATS-ML** training code, please cite this work as follows:

> R. He, “OATS-MCS: SCOPF Monte Carlo Sampling Framework,” Zenodo, 2026. [Online]. Available: https://doi.org/10.5281/zenodo.18792808

------

*This implementation builds upon the [OATS](https://github.com/bukhsh/oats) framework developed by Dr. Waqquas Bukhsh at the University of Strathclyde.*

