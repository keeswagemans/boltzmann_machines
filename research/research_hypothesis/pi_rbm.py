import numpy as np
import matplotlib.pyplot as plt
import os

class GaussianRBM:
    def __init__(self, n_visible, n_hidden, lr=0.001, sigma=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.sigma = sigma
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_hidden(self, v):
        activation = np.dot(v, self.W) / (self.sigma + 1e-8) + self.h_bias
        h_prob = self._sigmoid(activation)
        h_state = (h_prob > np.random.rand(*h_prob.shape)).astype(np.float32)
        return h_prob, h_state

    def sample_visible(self, h):
        v_mean = self.sigma * np.dot(h, self.W.T) + self.v_bias
        v_sample = v_mean + np.random.randn(*v_mean.shape) * self.sigma
        return v_mean, v_sample

    def train_step(self, v0, mask, lambda_phys=0.01, alpha=0.01, nx=20, nt=20, dx=0.05, dt=0.05):
        batch_size = v0.shape[0]
        
        h0_prob, h0_state = self.sample_hidden(v0)
        v1_mean, v1_sample = self.sample_visible(h0_state)
        h1_prob, _ = self.sample_hidden(v1_sample)
        
        dW = (np.dot(v0.T, h0_prob) - np.dot(v1_sample.T, h1_prob)) / (batch_size * self.sigma)
        dv_bias = np.mean(v0 - v1_sample, axis=0) / (self.sigma**2)
        dh_bias = np.mean(h0_prob - h1_prob, axis=0)
        
        if lambda_phys > 0:
            V = v1_mean.reshape(batch_size, nx, nt)
            u_t = (V[:, 1:-1, 1:] - V[:, 1:-1, :-1]) / dt
            u_xx = (V[:, 2:, :-1] - 2*V[:, 1:-1, :-1] + V[:, :-2, :-1]) / (dx**2)
            res = u_t - alpha * u_xx
            
            grad_V = np.zeros_like(V)
            grad_V[:, 1:-1, 1:] += 2 * res / dt
            grad_V[:, 1:-1, :-1] -= 2 * res / dt
            grad_V[:, 2:, :-1] -= 2 * res * alpha / (dx**2)
            grad_V[:, 1:-1, :-1] += 2 * res * 2 * alpha / (dx**2)
            grad_V[:, :-2, :-1] -= 2 * res * alpha / (dx**2)

            grad_v_phys = grad_V.reshape(batch_size, -1)
            dW_phys = self.sigma * np.dot(grad_v_phys.T, h0_prob) / batch_size
            dv_bias_phys = np.mean(grad_v_phys, axis=0)
            
            dW -= lambda_phys * dW_phys
            dv_bias -= lambda_phys * dv_bias_phys

        # Gradient clipping
        np.clip(dW, -1, 1, out=dW)
        np.clip(dv_bias, -1, 1, out=dv_bias)
        np.clip(dh_bias, -1, 1, out=dh_bias)

        self.W += self.lr * dW
        self.v_bias += self.lr * dv_bias
        self.h_bias += self.lr * dh_bias
        
        error = np.mean(((v0 - v1_sample) * mask)**2)
        return error

    def reconstruct(self, v_init, mask, steps=10):
        v = v_init.copy()
        for _ in range(steps):
            h_prob, h_state = self.sample_hidden(v)
            v_mean, v_sample = self.sample_visible(h_state)
            # Keep the known data
            v = v * mask + v_sample * (1 - mask)
        return v_mean

if __name__ == "__main__":
    # Load data
    data = np.load('data/heat_data.npz')
    nx, nt = 20, 20
    U_true = data['U_true']
    
    # Create mask and initial vector
    mask = np.zeros((nx, nt))
    v_init = np.zeros((nx, nt))
    
    # This is slightly tricky: we need to map sparse (x, t) to grid indices
    x_grid = data['x_grid']
    t_grid = data['t_grid']
    
    for i in range(len(data['x_train'])):
        xi = np.argmin(np.abs(x_grid - data['x_train'][i]))
        ti = np.argmin(np.abs(t_grid - data['t_train'][i]))
        mask[xi, ti] = 1.0
        v_init[xi, ti] = data['u_train'][i]
        
    v0 = v_init.flatten().reshape(1, -1)
    m0 = mask.flatten().reshape(1, -1)
    
    rbm = GaussianRBM(n_visible=nx*nt, n_hidden=100, lr=0.01)
    
    for epoch in range(2000):
        error = rbm.train_step(v0, m0)
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Error: {error:.6f}")
            
    # Reconstruction
    v_rec = rbm.reconstruct(v0, m0, steps=50).reshape(nx, nt)
    
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(data['T'], data['X'], v_rec, shading='auto')
    plt.colorbar(label='u_rec')
    plt.title('RBM Reconstruction (Data Only)')
    plt.savefig('data/rbm_reconstruction.png')
    print("Reconstruction saved to data/rbm_reconstruction.png")
