import numpy as np
import matplotlib.pyplot as plt
import os
from pinn_baseline import SimplePINN
from pi_rbm import GaussianRBM
from pi_bm import GaussianClassicBM
from pi_crbm import GaussianCRBM
from pi_dbn import GaussianDBN
from pi_dbm import GaussianDBM

def get_residual(u, alpha=0.01, dx=0.05, dt=0.05):
    """Calculates the mean squared physical residual for the 1D Heat Equation."""
    u_t = (u[:, 1:] - u[:, :-1]) / dt
    u_xx = (u[2:, :-1] - 2*u[1:-1, :-1] + u[:-2, :-1]) / (dx**2)
    return np.mean((u_t[1:-1, :] - alpha * u_xx)**2) 

def evaluate(): 
    # 1. Load Data
    data_path = 'data/heat_data.npz'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please run data_gen.py first.")
        
    data = np.load(data_path)
    nx, nt = 20, 20
    U_true = data['U_true']
    X, T = data['X'], data['T']
    
    x_grid = data['x_grid']
    t_grid = data['t_grid']
    mask = np.zeros((nx, nt))
    v_init = np.zeros((nx, nt))
    
    for i in range(len(data['x_train'])):
        xi = np.argmin(np.abs(x_grid - data['x_train'][i]))
        ti = np.argmin(np.abs(t_grid - data['t_train'][i]))
        mask[xi, ti] = 1.0
        v_init[xi, ti] = data['u_train'][i]
        
    v0 = v_init.flatten().reshape(1, -1)
    m0 = mask.flatten().reshape(1, -1)
    
    epochs = 1000
    sigma = 0.3
    
    # 2. Train PINN
    print("\nTraining PINN Baseline", flush=True)
    pinn = SimplePINN()
    for epoch in range(1000):
        pinn.train_step(data['x_train'], data['t_train'], data['u_train'], None, None)
    
    # PINN Prediction
    X_grid_flat = np.stack([X.flatten(), T.flatten()], axis=1)
    u_pinn = pinn.forward(X_grid_flat).reshape(nx, nt)
    pinn_mse = np.mean((u_pinn - U_true)**2)
    pinn_res = get_residual(u_pinn)
    
    # 3. Train all Boltzmann models
    models = {
        'PI-RBM': GaussianRBM(n_visible=nx*nt, n_hidden=100, lr=0.001, sigma=sigma),
        'PI-ClassicBM': GaussianClassicBM(n_visible=nx*nt, n_hidden=50, lr=0.001, sigma=sigma),
        'PI-CRBM': GaussianCRBM(input_shape=(nx, nt), filter_shape=(3, 3), n_filters=4, lr=0.001, sigma=sigma),
        'PI-DBN': GaussianDBN(layer_sizes=[nx*nt, 100, 50], lr=0.001, sigma=sigma),
        'PI-DBM': GaussianDBM(n_visible=nx*nt, n_h1=50, n_h2=20, lr=0.001, sigma=sigma)
    }
    
    # Pre-train DBN bottom layer
    models['PI-DBN'].pretrain(v0, epochs=10, batch_size=1)
    
    # Training Loop
    for name, model in models.items():
        for epoch in range(epochs):
            model.train_step(v0, m0, lambda_phys=0.001)
            
    # 4. Reconstruct & Evaluate
    n_samples = 30
    eval_results = {}
    
    for name, model in models.items():
        samples = []
        for _ in range(n_samples):
            v_rec = model.reconstruct(v0, m0, steps=30).reshape(nx, nt)
            samples.append(v_rec)
        samples = np.array(samples)
        
        mean_field = np.mean(samples, axis=0)
        std_field = np.std(samples, axis=0)
        
        mse = np.mean((mean_field - U_true)**2)
        res = get_residual(mean_field)
        within_2sigma = (U_true >= mean_field - 2*std_field) & (U_true <= mean_field + 2*std_field)
        coverage = np.mean(within_2sigma)
        
        eval_results[name] = {
            'mean': mean_field,
            'std': std_field,
            'mse': mse,
            'res': res,
            'coverage': coverage
        }

    # 5. Output Markdown Results Table
    print("BENCHMARK COMPARISON RESULTS")
    print(f"| Model | MSE | Physical Residual | 2-Sigma UQ Coverage |")
    print(f"| PINN Baseline | {pinn_mse:.6f} | {pinn_res:.6f} | N/A (Deterministic) |")
    for name in models.keys():
        res = eval_results[name]
        print(f"| {name} | {res['mse']:.6f} | {res['res']:.6f} | {res['coverage']:.2%} |")

    # 6. Plotting comprehensive grid
    plt.figure(figsize=(18, 14))
    
    # True Solution
    plt.subplot(4, 4, 1)
    plt.pcolormesh(T, X, U_true, shading='auto', cmap='viridis')
    plt.colorbar()
    plt.scatter(data['t_train'], data['x_train'], c='red', s=10, alpha=0.6, label='Observations')
    plt.title('True Solution (Heat Eq)')
    plt.legend()
    
    # PINN
    plt.subplot(4, 4, 2)
    plt.pcolormesh(T, X, u_pinn, shading='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f'PINN (MSE: {pinn_mse:.4f})')
    
    # PINN Error
    plt.subplot(4, 4, 3)
    plt.pcolormesh(T, X, np.abs(u_pinn - U_true), shading='auto', cmap='inferno')
    plt.colorbar()
    plt.title('PINN Absolute Error')
    
    plot_idx = 4
    for name in ['PI-RBM', 'PI-ClassicBM', 'PI-CRBM', 'PI-DBN', 'PI-DBM']:
        res = eval_results[name]
        
        # Mean Reconstructed Field
        plt.subplot(4, 4, plot_idx)
        plt.pcolormesh(T, X, res['mean'], shading='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'{name} Mean (MSE: {res["mse"]:.4f})')
        plot_idx += 1
        
        # Uncertainty Field (Standard Dev)
        plt.subplot(4, 4, plot_idx)
        plt.pcolormesh(T, X, res['std'], shading='auto', cmap='magma')
        plt.colorbar()
        plt.title(f'{name} Uncertainty (Std)')
        plot_idx += 1
        
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plot_save_path = 'data/final_comparison.png'
    plt.savefig(plot_save_path, dpi=150)

if __name__ == "__main__":
    evaluate()
