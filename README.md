# Hybrid Optimization for Congestion Management in Deregulated Power Systems

[![Conference](https://img.shields.io/badge/Conference-IEEE%20EPREC%202026-blue.svg)](https://eprec.co.in)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FEPREC66546.2026.11412040-darkgreen.svg)](https://doi.org/10.1109/EPREC66546.2026.11412040)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation and experimental codebase for the research paper:

> **"Hybrid Optimization for Congestion Management in Deregulated Power Systems"**  
> *Presented at the 6th International Conference on Electric Power and Renewable Energy (EPREC-2026)*  
> *Organized by the Department of Electrical Engineering, Indian Institute of Technology (IIT) Bhilai, India.*  
> **Publisher:** IEEE  
> **DOI:** [10.1109/EPREC66546.2026.11412040](https://doi.org/10.1109/EPREC66546.2026.11412040)

---

## 📌 Overview

In modern deregulated power markets, transmission open access and variable renewable generation frequently induce **transmission line congestion**. Operating lines beyond their thermal and transfer limits threatens grid security and risks cascading blackouts.

This repository provides an end-to-end framework for **optimal congestion management** that strategically redispatches active power generation and performs minimal, controlled load shedding to relieve transmission line overloads while minimizing rescheduling costs and operational disruptions.

---

## 🔬 Key Contributions & Methodology

### 1. Sensitivity Modeling (PTDF / DC Power Flow)
The relationship between nodal active power injections and line power flows is modeled via the Power Transfer Distribution Factor (PTDF) sensitivity matrix:
$$\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$$
where:
- $\mathbf{B} \in \mathbb{R}^{N \times 1}$ is the bus active power injection vector.
- $\mathbf{A} \in \mathbb{R}^{M \times N}$ is the PTDF sensitivity matrix.
- $\mathbf{C} \in \mathbb{R}^{M \times 1}$ represents the active power flows across all $M$ transmission lines.

### 2. Multi-Objective Hierarchical Formulation
The objective function strictly prioritizes grid security and consumer service:
1. **Hard Constraints (Penalty Multiplier $10^{10}$):**
   - Transmission line thermal capacity limits: $|C_i| \le C_{i,\max}$
   - Generator physical limits: $P_{Gi,\min} \le P_{Gi} \le P_{Gi,\max}$
   - Load sheddability constraints: $P_{Li} \le 0$ (loads cannot inject power)
   - Real power balance constraint: $\sum B_i = 0$
2. **Primary Operational Objective (Weight $10^7$):**
   - Strictly minimize total shedded load $\sum (P_{Li} - P_{Li,\text{init}})^2$.
3. **Secondary Operational Objective (Weight $10^5$):**
   - Minimize total deviation of generators from baseline schedules: $\sum |P_{Gi} - P_{Gi,\text{init}}|$.
4. **Economic Objective (Weight $0.1$):**
   - Minimize generator rescheduling cost: $\sum C_i \cdot |P_{Gi} - P_{Gi,\text{init}}|$.

### 3. Novel Tri-Hybrid Metaheuristic Optimizer
To avoid premature convergence in high-dimensional non-convex search spaces, the framework couples three nature-inspired optimization algorithms with an **adaptive probability distribution**:
- **Orcas Optimization Algorithm (OOA):** Exploits promising solution spaces via time-decaying coordinated pursuit steps.
- **Krill Herd Algorithm (KHA):** Simulates krill herding and foraging behavior to provide extensive global exploration.
- **Spotted Hyena Optimizer (SHO):** Uses social hierarchy mechanisms to encircle prey, balancing exploration and exploitation.
- **Adaptive Probability Updating:** Dynamically boosts the probability of selecting whichever algorithm yields the highest success rate in reducing fitness.
- **Stagnation Handling & Memory Diversification:** Replaces the worst population fraction with diverse candidates when fitness plateaus.
- **Local Search Operator:** Applies fine-grained perturbations to fine-tune individual bus injections.
- **Hybrid Polish (Two-Stage Solver):** Employs global metaheuristic exploration followed by deterministic local optimization (SciPy SLSQP / interior-point) to reach high precision.

---

## 📂 Project Structure

```
├── README.md                           # Documentation & research overview
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules
├── BusSys.py                           # Plotly interactive 30-bus network graph visualization
├── bus_link.py                         # NetworkX and Matplotlib schematic layout
├── ReadData.py                         # Data ingestion for loads and generator limits
├── con_1.ipynb                         # Interactive Jupyter notebook for system inspection
├── generator_data.csv                  # Generator capacity limits (Pmin, Pmax, Qmin, Qmax)
├── load_bus_shunt_capacitor_data.csv   # Bus real/reactive loads and shunt capacitor data
│
├── final/                              # Production-ready implementation
│   ├── samp.py                         # Standalone interactive CLI optimizer application
│   ├── this_is_it.ipynb                # Interactive Jupyter notebook implementation
│   ├── lock.csv                        # Validated PTDF sensitivity matrix (IEEE 14-bus)
│   ├── case_1.txt                      # Benchmark Scenario 1 (Overload relief audit)
│   ├── case_2.txt                      # Benchmark Scenario 2
│   └── case_3.txt                      # Benchmark Scenario 3
│
├── PowerFlowAnalyais/                  # AC Power Flow & Network Analysis
│   ├── PowerFlowAnalysis.py            # AC Newton-Raphson power flow using pandapower
│   ├── tint.py                         # AC power flow solver using PyPSA
│   └── 587.py                          # Early prototype of hybrid metaheuristic optimizer
│
└── algo/                               # Algorithmic Research & Telemetry
    ├── 14_bus_hyb-pol_3D-map.ipynb     # Two-stage Hybrid Polish + 3D PCA trajectory mapping
    ├── CSV_llm.ipynb                   # LLM natural language interface (TinyLlama) for log query
    ├── bolu.py                         # Adaptive weight metaheuristic benchmark
    └── optimization_log_*.csv          # Iteration-by-iteration telemetry logs
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/niranjan-crypt/Hybrid-Optimization-for-Congestion-Management-in-Deregulated-Power-Systems.git
cd Hybrid-Optimization-for-Congestion-Management-in-Deregulated-Power-Systems
pip install -r requirements.txt
```

### 2. Run the Congestion Management Optimizer
Execute the standalone CLI optimizer in `final/`:
```bash
cd final
python samp.py
```
Follow the interactive prompts to load the system matrix, specify line limits, and input initial bus injections.

### 3. Run AC Power Flow Analysis
```bash
python PowerFlowAnalyais/PowerFlowAnalysis.py
```

### 4. Interactive Visualization & Notebooks
Launch Jupyter to explore the interactive visual analytics:
```bash
jupyter notebook final/this_is_it.ipynb
```

---

## 📊 Experimental Results (IEEE 14-Bus Test System)

In **Case 1**, the IEEE 14-bus system was subjected to transmission congestion with Line 13 overloaded:

| Metric | Unoptimized (Pre-Optimization) | Optimized (Proposed Hybrid Optimizer) | Status |
| :--- | :---: | :---: | :---: |
| **Line 13 Flow** | **17.03 MW** (Limit: 15.00 MW) | **15.01 MW** | **Congestion Cleared** ✅ |
| **Line 5 Flow** | 40.91 MW (Limit: 50.00 MW) | 50.01 MW | Safe utilization ✅ |
| **Total Load Shed** | 0.00 MW | **3.39 MW** (98.6% demand preserved) | Negligible disruption ✅ |
| **Generator Deviation** | 0.00 MW | 246.55 MW | Re-dispatched safely ✅ |
| **System Feasibility** | **Violated (Infeasible)** | **Fully Feasible** | **Optimal** ✅ |

---

## 📖 Citation

If you find this work or code useful in your research, please cite our paper:

```bibtex
@inproceedings{eprec2026_congestion_mgmt,
  author    = {Krishnakumar, Niranjan and others},
  title     = {Hybrid Optimization for Congestion Management in Deregulated Power Systems},
  booktitle = {Proceedings of the 6th International Conference on Electric Power and Renewable Energy (EPREC-2026)},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/EPREC66546.2026.11412040},
  address   = {IIT Bhilai, India}
}
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
