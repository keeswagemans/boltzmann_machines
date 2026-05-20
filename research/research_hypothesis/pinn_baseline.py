import numpy as np
import matplotlib.pyplot as plt
import os

class SimplePINN:
    def __init__(self, layers=[2, 20, 20, 1], lr=0.01, alpha=0.01):
        self.layers = layers
        self.lr = lr
        self.alpha = alpha
        self.params = {}
        for i in range(len(layers) - 1):
            self.params[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            self.params[f'b{i+1}'] = np.zeros((1, layers[i+1]))

    def forward(self, X):
        self.a = {0: X}
        self.z = {}
        for i in range(len(self.layers) - 2):
            self.z[i+1] = np.dot(self.a[i], self.params[f'W{i+1}']) + self.params[f'b{i+1}']
            self.a[i+1] = np.tanh(self.z[i+1])
        
        i = len(self.layers) - 1
        self.z[i] = np.dot(self.a[i-1], self.params[f'W{i}']) + self.params[f'b{i}']
        self.a[i] = self.z[i]  # Linear output
        return self.a[i]

    def gradients(self, x, t):
        # Compute u_x, u_t, u_xx using manual backprop/differentiation
        # This is complex in numpy for a general MLP. 
        # For simplicity, let's use finite differences for the physics loss 
        # but exact gradients for the data loss.
        # Actually, for PINN to work well, we really want exact gradients.
        pass

    def train_step(self, x_data, t_data, u_data, x_phys, t_phys):
        # For this prototype, I'll use a simpler approach:
        # Just use data loss first to see if it can reconstruct.
        # Then we'll add physics.
        
        # Forward pass
        X = np.stack([x_data, t_data], axis=1)
        u_pred = self.forward(X).flatten()
        
        # Data Loss
        loss_data = np.mean((u_pred - u_data)**2)
        
        # Backprop (data only for now)
        # dL/du_pred = 2/N * (u_pred - u_data)
        grad_u = (u_pred - u_data).reshape(-1, 1) * (2.0 / len(u_data))
        
        # Gradient for W3, b3
        grad_W3 = np.dot(self.a[2].T, grad_u)
        grad_b3 = np.sum(grad_u, axis=0, keepdims=True)
        
        # Gradient for a2
        grad_a2 = np.dot(grad_u, self.params['W3'].T)
        grad_z2 = grad_a2 * (1 - self.a[2]**2)
        
        grad_W2 = np.dot(self.a[1].T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        
        # Gradient for a1
        grad_a1 = np.dot(grad_z2, self.params['W2'].T)
        grad_z1 = grad_a1 * (1 - self.a[1]**2)
        
        grad_W1 = np.dot(self.a[0].T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
        
        # Update
        self.params['W3'] -= self.lr * grad_W3
        self.params['b3'] -= self.lr * grad_b3
        self.params['W2'] -= self.lr * grad_W2
        self.params['b2'] -= self.lr * grad_b2
        self.params['W1'] -= self.lr * grad_W1
        self.params['b1'] -= self.lr * grad_b1
        
        return loss_data

if __name__ == "__main__":
    # Load data
    data = np.load('data/heat_data.npz')
    x_train = data['x_train']
    t_train = data['t_train']
    u_train = data['u_train']
    
    pinn = SimplePINN()
    
    for epoch in range(1000):
        loss = pinn.train_step(x_train, t_train, u_train, None, None)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
    # Evaluation
    X_grid = np.stack([data['X'].flatten(), data['T'].flatten()], axis=1)
    u_pred = pinn.forward(X_grid).reshape(data['X'].shape)
    
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(data['T'], data['X'], u_pred, shading='auto')
    plt.colorbar(label='u_pred')
    plt.title('PINN Reconstruction (Data Only)')
    plt.savefig('data/pinn_reconstruction.png')
    print("Reconstruction saved to data/pinn_reconstruction.png")
