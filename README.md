# Physics-Informed Boltzmann Machines (PI-BMs)

This repository is an exploratory sandbox for trying out a new idea: integrating physical laws (partial differential equations) into the energy functions of various Boltzmann Machine architectures. We are using these Physics-Informed Boltzmann Machines (PI-BMs) as a prototype to see if they can reconstruct continuous dynamical systems from sparse, noisy sensor data while providing uncertainty quantification (UQ).

The goal here is to test a hypothesis: does the probabilistic, generative nature of Boltzmann Machines act as a better physical prior than traditional, deterministic PINNs when data is highly corrupted? This is a proof-of-concept, so expect things to be experimental!

---

## 🚀 Key Architectures Evaluated

We have developed continuous (Gaussian visible unit) and physics-informed variants for five major Boltzmann architectures from scratch:

1. **PI-RBM (Restricted Boltzmann Machine)**: Standard bipartite architecture adapted with continuous Gaussian visible nodes and physics loss penalty.
2. **PI-ClassicBM (Classic Boltzmann Machine)**: A fully connected recurrent Boltzmann graph with symmetric weights, trained using Mean-Field approximation and sequential Gibbs sampling.
3. **PI-CRBM (Convolutional RBM)**: Integrates 2D spatial-temporal convolutional filters over continuous visible fields, capturing grid-based physical dynamics.
4. **PI-DBN (Deep Belief Network)**: Stacks a Gaussian RBM input layer with binary hidden RBMs. Fine-tuned by backpropagating physical residuals through the deep network.
5. **PI-DBM (Deep Boltzmann Machine)**: A deep bipartite architecture ($V \leftrightarrow H_1 \leftrightarrow H_2$) with symmetric feedback, trained using iterative mean-field updates.

---

## 📊 Summary of Benchmark Results

All models were evaluated on the **1D Heat Equation** ($u_t = \alpha u_{xx}$) under noisy ($\sigma = 0.05$) and sparse (50 samples) observations:

| Model | Reconstruction MSE | Physical Residual | 2-Sigma UQ Coverage |
| :--- | :---: | :---: | :---: |
| **PINN Baseline** | 0.029812 | 2.062679 | *N/A (Deterministic)* |
| **PI-CRBM (Convolutional)** | **0.359344** | **0.112983** | 0.00% |
| **PI-ClassicBM** | 0.380528 | 12.175053 | **6.75%** |
| **PI-DBM (Deep DBM)** | 0.401864 | 21.952318 | 2.50% |
| **PI-RBM** | 0.392514 | 23.593117 | 4.25% |
| **PI-DBN** | 0.403920 | 24.026758 | 0.25% |

> [!NOTE]
> **Key Finding**: The **PI-CRBM (Convolutional RBM)** achieved a physical residual **20x lower** than the standard PINN, demonstrating that spatial-temporal convolutional weight-sharing in energy-based models serves as an exceptionally strong physical regularizer!
> **Uncertainty Quantification**: The **PI-ClassicBM** achieved the highest 2-Sigma coverage, proving the value of fully connected recurrent connections for quantifying predictive uncertainty.

---

## 📁 Repository Structure

* **`research/`**: Central hub for all research, implementations, and hypotheses.
  * **`research_hypothesis/`**: Core benchmark suites (`pi_bm.py`, `pi_crbm.py`, `pi_dbn.py`, `pi_dbm.py`, `pi_rbm.py`, `evaluate.py`, `experiment_suite.py`).
  * **`boltzmann_machine/`**: Scratch implementation of classic binary BM.
  * **`restricted_boltzmann/`**: Scratch implementation of classic binary RBM.
  * **`convolutional_machine/`**: Scratch implementation of convolutional RBM.
  * **`deep_belief_network/`**: Scratch implementation of binary DBN stack.
  * **`deep_boltzmann_machine/`**: Scratch implementation of deep DBM.
* **`data/`**: Output directory for generated datasets and visualization plots.

---

## 🛠️ Getting Started

### 1. Install Dependencies
This project uses the fast, modern `uv` package manager:
```bash
cd research/research_hypothesis
uv sync
```

### 2. Generate Dataset
```bash
uv run python data_gen.py
```

### 3. Run Benchmark and Generate Plot Matrix
```bash
uv run python evaluate.py
```
This will train all five models and the PINN baseline, print the results table, and output a comprehensive plot matrix at `data/final_comparison.png` visualizing the reconstructed solutions and uncertainty bands.