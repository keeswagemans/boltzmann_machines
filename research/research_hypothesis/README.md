# Research Hypothesis: Physics-Informed RBMs for PDE Reconstruction

## Hypothesis
Physics-informed RBMs will not necessarily beat PINNs on clean forward PDE solving, but may outperform them in uncertainty-aware reconstruction from sparse and noisy observations.

## Minimum Viable Paper Setup

### 1. Data Generation
* Generate sparse and noisy PDE observations (e.g., Heat equation, Burger's equation).

### 2. PINN Baseline
* Train a standard Physics-Informed Neural Network (PINN) on the sparse/noisy observations.
* Evaluate reconstruction accuracy and physical residual.

### 3. Physics-Informed RBM (PI-RBM)
* Train a physics-informed RBM on the same sparse/noisy data.
* Incorporate physical constraints into the RBM energy function or training objective.

### 4. Comparison and Evaluation
* **Reconstruction Accuracy:** Compare mean reconstruction error against PINN.
* **Physical Residual:** Evaluate how well the reconstructions satisfy the PDE.
* **Uncertainty Quantification:** 
    * Sample multiple fields from the trained RBM.
    * Evaluate whether the true solution lies within the predicted uncertainty bands (e.g., 95% confidence interval).
* **Robustness:** PINNs often struggle with noise or very sparse data without heavy regularization; evaluate if the generative nature of RBMs provides better priors.

### 5. Ablation Studies
* **No physics penalty:** How much does the physical constraint help?
* **Varying data noise:** Performance as a function of SNR.
* **Varying sensor density:** Performance as a function of observation sparsity.
