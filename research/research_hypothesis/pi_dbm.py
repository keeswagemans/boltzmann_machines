import numpy as np

class GaussianDBM:
    def __init__(self, n_visible, n_h1, n_h2, lr=0.001, sigma=0.1, random_state=42):
        """
        Physics-Informed Gaussian Deep Boltzmann Machine (PI-GDBM).
        Symmetric architecture: V <-> H1 <-> H2.
        Visible units V are continuous (Gaussian).
        Hidden units H1, H2 are binary (Bernoulli).
        """
        self.n_v = n_visible
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.lr = lr
        self.sigma = sigma
        
        rng = np.random.default_rng(random_state)
        # Weight matrices
        self.W1 = rng.normal(0.0, 0.01, (n_visible, n_h1))
        self.W2 = rng.normal(0.0, 0.01, (n_h1, n_h2))
        
        # Biases
        self.v_bias = np.zeros(n_visible)
        self.h1_bias = np.zeros(n_h1)
        self.h2_bias = np.zeros(n_h2)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_h1(self, v, h2):
        activation = np.dot(v, self.W1) / self.sigma + np.dot(h2, self.W2.T) + self.h1_bias
        prob = self._sigmoid(activation)
        state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
        return prob, state

    def sample_v(self, h1):
        v_mean = self.sigma * np.dot(h1, self.W1.T) + self.v_bias
        v_sample = v_mean + np.random.randn(*v_mean.shape) * self.sigma
        return v_mean, v_sample

    def sample_h2(self, h1):
        activation = np.dot(h1, self.W2) + self.h2_bias
        prob = self._sigmoid(activation)
        state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
        return prob, state

    def train_step(self, v0, mask, lambda_phys=0.01, alpha=0.01, nx=20, nt=20, dx=0.05, dt=0.05, n_gibbs=5):
        batch_size = v0.shape[0]
        
        # 1. Positive Phase: Mean Field Approximation
        mu1 = np.full((batch_size, self.n_h1), 0.5)
        mu2 = np.full((batch_size, self.n_h2), 0.5)
        
        for _ in range(10):
            mu1 = self._sigmoid(np.dot(v0, self.W1) / self.sigma + np.dot(mu2, self.W2.T) + self.h1_bias)
            mu2 = self._sigmoid(np.dot(mu1, self.W2) + self.h2_bias)
            
        # 2. Negative Phase: Gibbs Sampling (Stochastic Approximation)
        v_neg = v0.copy()
        h1_neg = (np.random.rand(batch_size, self.n_h1) > 0.5).astype(np.float32)
        h2_neg = (np.random.rand(batch_size, self.n_h2) > 0.5).astype(np.float32)
        
        for _ in range(n_gibbs):
            _, h1_neg = self.sample_h1(v_neg, h2_neg)
            v_neg_mean, v_neg = self.sample_v(h1_neg)
            _, h2_neg = self.sample_h2(h1_neg)
            
        # Compute expectations & standard CD/Boltzmann gradients
        dW1 = (np.dot(v0.T, mu1) - np.dot(v_neg.T, h1_neg)) / (batch_size * self.sigma)
        dW2 = (np.dot(mu1.T, mu2) - np.dot(h1_neg.T, h2_neg)) / batch_size
        
        dv_bias = np.mean(v0 - v_neg, axis=0) / (self.sigma**2)
        dh1_bias = np.mean(mu1 - h1_neg, axis=0)
        dh2_bias = np.mean(mu2 - h2_neg, axis=0)
        
        # 3. Physics penalty on positive reconstruction mean
        v_pos_mean = self.sigma * np.dot(mu1, self.W1.T) + self.v_bias
        
        if lambda_phys > 0:
            V = v_pos_mean.reshape(batch_size, nx, nt)
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
            
            dW1_phys = self.sigma * np.dot(grad_v_phys.T, mu1) / batch_size
            dv_bias_phys = np.mean(grad_v_phys, axis=0)
            
            dW1 -= lambda_phys * dW1_phys
            dv_bias -= lambda_phys * dv_bias_phys

        # Gradient clipping
        np.clip(dW1, -1, 1, out=dW1)
        np.clip(dW2, -1, 1, out=dW2)
        np.clip(dv_bias, -1, 1, out=dv_bias)
        np.clip(dh1_bias, -1, 1, out=dh1_bias)
        np.clip(dh2_bias, -1, 1, out=dh2_bias)

        self.W1 += self.lr * dW1
        self.W2 += self.lr * dW2
        self.v_bias += self.lr * dv_bias
        self.h1_bias += self.lr * dh1_bias
        self.h2_bias += self.lr * dh2_bias
        
        error = np.mean(((v0 - v_neg) * mask)**2)
        return error

    def reconstruct(self, v_init, mask, steps=10):
        batch_size = v_init.shape[0]
        v = v_init.copy()
        h1 = (np.random.rand(batch_size, self.n_h1) > 0.5).astype(np.float32)
        h2 = (np.random.rand(batch_size, self.n_h2) > 0.5).astype(np.float32)
        
        for _ in range(steps):
            _, h1 = self.sample_h1(v, h2)
            v_mean, v_sample = self.sample_v(h1)
            _, h2 = self.sample_h2(h1)
            v = v_init * mask + v_sample * (1 - mask)
            
        return v_mean
