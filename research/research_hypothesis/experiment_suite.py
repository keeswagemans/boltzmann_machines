import numpy as np
from data_gen import generate_heat_data
from pinn_baseline import SimplePINN
from pi_rbm import GaussianRBM
from pi_bm import GaussianClassicBM
from pi_crbm import GaussianCRBM
from pi_dbn import GaussianDBN
from pi_dbm import GaussianDBM
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
    
    epochs_bm = 1000  # Set to a fast but converging value
    
    # 2. Train Models
    # RBM (Gaussian RBM)
    rbm = GaussianRBM(n_visible=nx*nt, n_hidden=100, lr=0.001, sigma=sigma)
    for epoch in range(epochs_bm):
        rbm.train_step(v0, m0, lambda_phys=lambda_phys)
        
    # Classic BM
    gbm = GaussianClassicBM(n_visible=nx*nt, n_hidden=50, lr=0.001, sigma=sigma)
    for epoch in range(epochs_bm):
        gbm.train_step(v0, m0, lambda_phys=lambda_phys)
        
    # CRBM
    crbm = GaussianCRBM(input_shape=(nx, nt), filter_shape=(3, 3), n_filters=4, lr=0.001, sigma=sigma)
    for epoch in range(epochs_bm):
        crbm.train_step(v0, m0, lambda_phys=lambda_phys)
        
    # DBN
    dbn = GaussianDBN(layer_sizes=[nx*nt, 100, 50], lr=0.001, sigma=sigma)
    dbn.pretrain(v0, epochs=10, batch_size=1)
    for epoch in range(epochs_bm):
        dbn.train_step(v0, m0, lambda_phys=lambda_phys)
        
    # DBM
    dbm = GaussianDBM(n_visible=nx*nt, n_h1=50, n_h2=20, lr=0.001, sigma=sigma)
    for epoch in range(epochs_bm):
        dbm.train_step(v0, m0, lambda_phys=lambda_phys)
        
    # PINN Baseline
    pinn = SimplePINN()
    for epoch in range(1000):
        pinn.train_step(data['x_train'], data['t_train'], data['u_train'], None, None)
        
    # 3. Evaluate PINN
    X_grid_flat = np.stack([data['X'].flatten(), data['T'].flatten()], axis=1)
    u_pinn = pinn.forward(X_grid_flat).reshape(nx, nt)
    pinn_mse = np.mean((u_pinn - U_true)**2)
    pinn_res = get_residual(u_pinn)
    
    # 4. Evaluate Boltzmann Models (Sampling/Uncertainty Reconstruction)
    n_samples_eval = 20
    results = {}
    
    models = {
        'RBM': rbm,
        'ClassicBM': gbm,
        'CRBM': crbm,
        'DBN': dbn,
        'DBM': dbm
    }
    
    for name, model in models.items():
        samples = []
        for _ in range(n_samples_eval):
            v_rec = model.reconstruct(v0, m0, steps=30).reshape(nx, nt)
            samples.append(v_rec)
        samples = np.array(samples)
        
        mean_field = np.mean(samples, axis=0)
        std_field = np.std(samples, axis=0)
        
        mse = np.mean((mean_field - U_true)**2)
        res = get_residual(mean_field)
        
        # Coverage
        within_2sigma = (U_true >= mean_field - 2*std_field) & (U_true <= mean_field + 2*std_field)
        coverage = np.mean(within_2sigma)
        
        results[name] = {
            'mse': mse,
            'res': res,
            'coverage': coverage
        }
        
    return {
        'pinn_mse': pinn_mse,
        'pinn_res': pinn_res,
        'bm_results': results
    }

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING COMPREHENSIVE PHYSICS-INFORMED BOLTZMANN MACHINE EXPERIMENTS")
    print("=" * 60)
    
    # Run a comparative ablation study with low noise and with/without physics constraints
    print("\n--- Ablation Study: Data-Only vs Physics-Informed (Noise = 0.05) ---")
    res_no_phys = run_single_setup(noise_std=0.05, lambda_phys=0.0, sigma=0.3)
    res_with_phys = run_single_setup(noise_std=0.05, lambda_phys=0.01, sigma=0.3)
    
    print(f"\nPINN Baseline | MSE: {res_no_phys['pinn_mse']:.5f} | Phys Res: {res_no_phys['pinn_res']:.5f}")
    print("-" * 75)
    print(f"{'Model':<12} | {'No-Phys MSE':<12} {'No-Phys Res':<12} | {'With-Phys MSE':<13} {'With-Phys Res':<13} {'Coverage':<8}")
    print("-" * 75)
    for model_name in ['RBM', 'ClassicBM', 'CRBM', 'DBN', 'DBM']:
        r_np = res_no_phys['bm_results'][model_name]
        r_wp = res_with_phys['bm_results'][model_name]
        print(f"{model_name:<12} | {r_np['mse']:.5f}      {r_np['res']:.5f}      | {r_wp['mse']:.5f}       {r_wp['res']:.5f}       {r_wp['coverage']:.1%}")
    print("=" * 75)

    print("\n--- Noise Sensitivity Analysis ---")
    for noise in [0.1, 0.3]:
        print(f"\nTesting noise_std = {noise}...")
        res = run_single_setup(noise_std=noise, lambda_phys=0.01, sigma=0.3)
        print(f"PINN Baseline | MSE: {res['pinn_mse']:.5f} | Phys Res: {res['pinn_res']:.5f}")
        for model_name, metrics in res['bm_results'].items():
            print(f"PI-{model_name:<9} | MSE: {metrics['mse']:.5f} | Phys Res: {metrics['res']:.5f} | 2-Sigma Coverage: {metrics['coverage']:.1%}")
        print("-" * 60)
