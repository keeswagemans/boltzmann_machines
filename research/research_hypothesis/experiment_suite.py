import numpy as np
from data_gen import generate_heat_data
from pi_rbm import GaussianRBM
from pinn_baseline import SimplePINN
import sys

def get_residual(u, alpha=0.01, dx=0.05, dt=0.05):
    """Calculates the mean squared physical residual for the 1D Heat Equation."""
    u_t = (u[:, 1:] - u[:, :-1]) / dt
    u_xx = (u[2:, :-1] - 2*u[1:-1, :-1] + u[:-2, :-1]) / (dx**2)
    return np.mean((u_t[1:-1, :] - alpha * u_xx)**2)

def run_single_setup(noise_std=0.05, lambda_phys=0.001, sigma=0.1, n_samples_data=50):
    # 1. Generate Data
    data = generate_heat_data(nx=20, nt=20, n_samples=n_samples_data, noise_std=noise_std)
    nx, nt = 20, 20
    U_true = data['U_true']
    
    mask = np.zeros((nx, nt))
    v_init = np.zeros((nx, nt))
    x_grid, t_grid = data['x_grid'], data['t_grid']
    
    for i in range(len(data['x_train'])):
        xi = np.argmin(np.abs(x_grid - data['x_train'][i]))
        ti = np.argmin(np.abs(t_grid - data['t_train'][i]))
        mask[xi, ti] = 1.0
        v_init[xi, ti] = data['u_train'][i]
        
    v0 = v_init.flatten().reshape(1, -1)
    m0 = mask.flatten().reshape(1, -1)
    
    # 2. Train RBM
    rbm = GaussianRBM(n_visible=nx*nt, n_hidden=100, lr=0.001, sigma=sigma)
    for epoch in range(3000): # 3k for faster iterations
        rbm.train_step(v0, m0, lambda_phys=lambda_phys)
        
    # 3. Train PINN
    pinn = SimplePINN()
    for epoch in range(1000):
        pinn.train_step(data['x_train'], data['t_train'], data['u_train'], None, None)
        
    # 4. Evaluate
    X_grid_flat = np.stack([data['X'].flatten(), data['T'].flatten()], axis=1)
    u_pinn = pinn.forward(X_grid_flat).reshape(nx, nt)
    
    n_samples_rbm = 50
    rbm_samples = []
    for _ in range(n_samples_rbm):
        # Increased steps to 50 for better mixing
        v_rec = rbm.reconstruct(v0, m0, steps=50).reshape(nx, nt)
        rbm_samples.append(v_rec)
        
    rbm_samples = np.array(rbm_samples)
    u_rbm_mean = np.mean(rbm_samples, axis=0)
    u_rbm_std = np.std(rbm_samples, axis=0)
    
    pinn_mse = np.mean((u_pinn - U_true)**2)
    rbm_mse = np.mean((u_rbm_mean - U_true)**2)
    
    within_2sigma = (U_true >= u_rbm_mean - 2*u_rbm_std) & (U_true <= u_rbm_mean + 2*u_rbm_std)
    coverage = np.mean(within_2sigma)
    
    pinn_res = get_residual(u_pinn)
    rbm_res = get_residual(u_rbm_mean)
    
    return {
        'pinn_mse': pinn_mse,
        'rbm_mse': rbm_mse,
        'pinn_res': pinn_res,
        'rbm_res': rbm_res,
        'coverage': coverage
    }

if __name__ == "__main__":
    print("===================================================================")
    print("Experiment 1: Hyperparameter Tuning (Sigma & Physics Weight)")
    print("Goal: Find parameters that yield non-zero coverage and lower MSE.")
    print("===================================================================")
    best_sig, best_l_phys = 0.1, 0.001
    best_cov = -1
    
    for sig in [0.1, 0.3, 0.5]:
        for l_phys in [0.0, 0.001, 0.01]:
            print(f"Testing sigma={sig}, lambda_phys={l_phys}...", flush=True)
            res = run_single_setup(noise_std=0.05, lambda_phys=l_phys, sigma=sig)
            print(f"  -> RBM MSE: {res['rbm_mse']:.4f} | RBM Phys Res: {res['rbm_res']:.4f} | 2-Sigma Cov: {res['coverage']:.2%}")
            
            if res['coverage'] > best_cov:
                best_cov = res['coverage']
                best_sig, best_l_phys = sig, l_phys

    print("\n===================================================================")
    print("Experiment 2: Ablation Study (Physics vs No Physics)")
    print("Goal: Quantify impact of the physics loss term on reconstruction.")
    print("===================================================================")
    print(f"Using tuned sigma=0.5", flush=True)
    
    res_no_phys = run_single_setup(noise_std=0.05, lambda_phys=0.0, sigma=0.5)
    res_with_phys = run_single_setup(noise_std=0.05, lambda_phys=0.01, sigma=0.5)
    
    print(f"[No Physics] RBM MSE: {res_no_phys['rbm_mse']:.4f} | RBM Res: {res_no_phys['rbm_res']:.4f}")
    print(f"[W/ Physics] RBM MSE: {res_with_phys['rbm_mse']:.4f} | RBM Res: {res_with_phys['rbm_res']:.4f}")

    print("\n===================================================================")
    print("Experiment 3: Noise Sensitivity (Low Noise vs High Noise)")
    print("Goal: Compare PINN vs PI-RBM degradation as observation noise increases.")
    print("===================================================================")
    for noise in [0.05, 0.2, 0.5]:
        print(f"\nTesting noise_std={noise}...", flush=True)
        res = run_single_setup(noise_std=noise, lambda_phys=0.01, sigma=0.5)
        print(f"  PINN MSE: {res['pinn_mse']:.4f} | PI-RBM MSE: {res['rbm_mse']:.4f}")
        print(f"  PINN Res: {res['pinn_res']:.4f} | PI-RBM Res: {res['rbm_res']:.4f}")
        print(f"  PI-RBM Cov: {res['coverage']:.2%}")
