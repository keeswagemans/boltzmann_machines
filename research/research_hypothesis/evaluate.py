import numpy as np
import matplotlib.pyplot as plt
import os
from pi_rbm import GaussianRBM
from pinn_baseline import SimplePINN

def evaluate():
    # Load data
    data = np.load('data/heat_data.npz')
    nx, nt = 20, 20
    U_true = data['U_true']
    X, T = data['X'], data['T']
    
    # Load/Re-train RBM
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
    
    rbm = GaussianRBM(n_visible=nx*nt, n_hidden=100, lr=0.001)
    for epoch in range(5000):
        rbm.train_step(v0, m0, lambda_phys=0.001)
        
    # Load/Re-train PINN
    pinn = SimplePINN()
    for epoch in range(1000):
        pinn.train_step(data['x_train'], data['t_train'], data['u_train'], None, None)
        
    # PINN Prediction
    X_grid_flat = np.stack([X.flatten(), T.flatten()], axis=1)
    u_pinn = pinn.forward(X_grid_flat).reshape(nx, nt)
    
    # RBM Samples
    n_samples = 50
    rbm_samples = []
    for _ in range(n_samples):
        v_rec = rbm.reconstruct(v0, m0, steps=20).reshape(nx, nt)
        rbm_samples.append(v_rec)
        
    rbm_samples = np.array(rbm_samples)
    u_rbm_mean = np.mean(rbm_samples, axis=0)
    u_rbm_std = np.std(rbm_samples, axis=0)
    
    # Errors
    pinn_mse = np.mean((u_pinn - U_true)**2)
    rbm_mse = np.mean((u_rbm_mean - U_true)**2)
    
    # Uncertainty
    within_2sigma = (U_true >= u_rbm_mean - 2*u_rbm_std) & (U_true <= u_rbm_mean + 2*u_rbm_std)
    coverage = np.mean(within_2sigma)
    
    # Physical residual
    def get_residual(u, alpha=0.01, dx=0.05, dt=0.05):
        u_t = (u[:, 1:] - u[:, :-1]) / dt
        u_xx = (u[2:, :-1] - 2*u[1:-1, :-1] + u[:-2, :-1]) / (dx**2)
        return np.mean((u_t[1:-1, :] - alpha * u_xx)**2)

    pinn_res = get_residual(u_pinn)
    rbm_res = get_residual(u_rbm_mean)
    
    print(f"PINN MSE: {pinn_mse:.6f}, Physics Residual: {pinn_res:.6f}")
    print(f"PI-RBM MSE: {rbm_mse:.6f}, Physics Residual: {rbm_res:.6f}")
    print(f"PI-RBM 2nd Sigma Coverage: {coverage:.2%}")
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.pcolormesh(T, X, U_true, shading='auto')
    plt.colorbar()
    plt.title('True Solution')
    
    plt.subplot(2, 3, 2)
    plt.pcolormesh(T, X, u_pinn, shading='auto')
    plt.colorbar()
    plt.title(f'PINN (MSE: {pinn_mse:.4f})')
    
    plt.subplot(2, 3, 3)
    plt.pcolormesh(T, X, u_rbm_mean, shading='auto')
    plt.colorbar()
    plt.title(f'PI-RBM Mean (MSE: {rbm_mse:.4f})')
    
    plt.subplot(2, 3, 4)
    plt.pcolormesh(T, X, np.abs(u_pinn - U_true), shading='auto')
    plt.colorbar()
    plt.title('PINN Absolute Error')
    
    plt.subplot(2, 3, 5)
    plt.pcolormesh(T, X, u_rbm_std, shading='auto')
    plt.colorbar()
    plt.title('PI-RBM Std (Uncertainty)')
    
    plt.subplot(2, 3, 6)
    # Check if true solution is within 2 sigma
    within_2sigma = (U_true >= u_rbm_mean - 2*u_rbm_std) & (U_true <= u_rbm_mean + 2*u_rbm_std)
    coverage = np.mean(within_2sigma)
    plt.pcolormesh(T, X, within_2sigma.astype(float), shading='auto', cmap='RdYlGn')
    plt.colorbar()
    plt.title(f'2nd Sigma Coverage: {coverage:.2%}')
    
    plt.tight_layout()
    plt.savefig('data/final_comparison.png')
    print("Final comparison saved to data/final_comparison.png")

if __name__ == "__main__":
    evaluate()
