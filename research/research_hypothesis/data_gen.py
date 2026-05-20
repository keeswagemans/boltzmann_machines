import numpy as np
import matplotlib.pyplot as plt
import os

def generate_heat_data(nx=20, nt=20, alpha=0.01, L=1.0, T=1.0, n_samples=50, noise_std=0.05, seed=42):
    """
    Generates sparse and noisy observations for the 1D Heat Equation.
    u_t = alpha * u_xx
    Solution: u(x, t) = sin(pi * x) * exp(-pi^2 * alpha * t)
    """
    np.random.seed(seed)
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    X, T_grid = np.meshgrid(x, t)
    
    # Analytical solution
    U_true = np.sin(np.pi * X) * np.exp(-np.pi**2 * alpha * T_grid)
    
    # Sample sparse points
    indices = np.random.choice(nx * nt, n_samples, replace=False)
    x_flat = X.flatten()
    t_flat = T_grid.flatten()
    u_flat = U_true.flatten()
    
    x_train = x_flat[indices]
    t_train = t_flat[indices]
    u_train = u_flat[indices]
    
    # Add noise
    u_train_noisy = u_train + np.random.normal(0, noise_std, u_train.shape)
    
    return {
        'x_grid': x,
        't_grid': t,
        'X': X,
        'T': T_grid,
        'U_true': U_true,
        'x_train': x_train,
        't_train': t_train,
        'u_train': u_train_noisy,
        'u_train_clean': u_train,
        'alpha': alpha,
        'noise_std': noise_std
    }

if __name__ == "__main__":
    data = generate_heat_data()
    
    # Save data
    os.makedirs('data', exist_ok=True)
    np.savez('data/heat_data.npz', **data)
    print(f"Data generated and saved to data/heat_data.npz")
    
    # Plotting for verification
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(data['T'], data['X'], data['U_true'], shading='auto')
    plt.colorbar(label='u(x,t)')
    plt.scatter(data['t_train'], data['x_train'], c='red', s=5, alpha=0.5, label='Samples')
    plt.title('True Solution and Sample Points')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(data['u_train'])), data['u_train'], s=2, label='Noisy')
    plt.scatter(range(len(data['u_train_clean'])), data['u_train_clean'], s=2, label='Clean')
    plt.title('Sampled Values')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/heat_data_vis.png')
    print("Visualization saved to data/heat_data_vis.png")
