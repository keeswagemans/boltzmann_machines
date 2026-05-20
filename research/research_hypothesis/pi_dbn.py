import numpy as np
from pi_rbm import GaussianRBM

class RestrictedBoltzmannMachineBinary:
    """Standard Binary-Binary RBM used for higher layers of DBN."""
    def __init__(self, n_visible, n_hidden, lr=0.1, random_state=42):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        rng = np.random.default_rng(random_state)
        self.W = rng.normal(loc=0.0, scale=0.01, size=(n_visible, n_hidden))
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_hidden(self, v):
        activation = np.dot(v, self.W) + self.h_bias
        h_prob = self._sigmoid(activation)
        h_state = (h_prob > np.random.rand(*h_prob.shape)).astype(np.float32)
        return h_prob, h_state

    def sample_visible(self, h):
        activation = np.dot(h, self.W.T) + self.v_bias
        v_prob = self._sigmoid(activation)
        v_state = (v_prob > np.random.rand(*v_prob.shape)).astype(np.float32)
        return v_prob, v_state

    def train_step(self, v0):
        batch_size = v0.shape[0]
        h0_prob, h0_state = self.sample_hidden(v0)
        
        # CD-1
        _, h1_state = self.sample_hidden(v0)
        v1_prob, v1_state = self.sample_visible(h1_state)
        h1_prob, _ = self.sample_hidden(v1_state)
        
        self.W += self.lr * (np.dot(v0.T, h0_prob) - np.dot(v1_state.T, h1_prob)) / batch_size
        self.v_bias += self.lr * np.mean(v0 - v1_state, axis=0)
        self.h_bias += self.lr * np.mean(h0_prob - h1_prob, axis=0)
        return np.mean((v0 - v1_state)**2)

class GaussianDBN:
    def __init__(self, layer_sizes, lr=0.001, sigma=0.1, random_state=42):
        """
        Physics-Informed Gaussian Deep Belief Network (PI-GDBN).
        Stacked RBMs where the bottom layer is Gaussian-Bernoulli and
        the higher layers are standard binary-binary RBMs.
        """
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.sigma = sigma
        
        self.rbms = []
        # First layer: Gaussian RBM
        self.rbms.append(GaussianRBM(n_visible=layer_sizes[0], n_hidden=layer_sizes[1], lr=lr, sigma=sigma))
        
        # Higher layers: Binary RBMs
        for i in range(1, len(layer_sizes) - 1):
            self.rbms.append(RestrictedBoltzmannMachineBinary(
                n_visible=layer_sizes[i],
                n_hidden=layer_sizes[i+1],
                lr=lr,
                random_state=random_state + i
            ))

    def pretrain(self, X, epochs=10, batch_size=32):
        """Greedy layer-wise unsupervised pre-training."""
        current_input = X
        mask_ones = np.ones_like(X) # Dummy mask for pre-training
        
        for i, rbm in enumerate(self.rbms):
            print(f"Pre-training Layer {i+1}: {rbm.n_visible} -> {rbm.n_hidden}")
            for epoch in range(epochs):
                indices = np.random.permutation(len(current_input))
                total_error = 0
                for start_idx in range(0, len(current_input), batch_size):
                    batch = current_input[indices[start_idx : start_idx + batch_size]]
                    if i == 0:
                        error = rbm.train_step(batch, mask_ones[indices[start_idx : start_idx + batch_size]], lambda_phys=0)
                    else:
                        error = rbm.train_step(batch)
                    total_error += error
            _, current_input = rbm.sample_hidden(current_input)

    def reconstruct(self, v_init, mask, steps=10):
        # Forward (Up)
        h_states = [v_init]
        for rbm in self.rbms:
            _, next_h = rbm.sample_hidden(h_states[-1])
            h_states.append(next_h)
            
        # Backward (Down) with Gibbs sampling iterations at top
        top_h = h_states[-1]
        for _ in range(steps):
            # Sample down and up at top layer if desired, or simple back-reconstruction
            pass
            
        current_recon = top_h
        for rbm in reversed(self.rbms):
            if isinstance(rbm, GaussianRBM):
                current_recon, _ = rbm.sample_visible(current_recon)
            else:
                current_recon, _ = rbm.sample_visible(current_recon)
                
        return current_recon

    def train_step(self, v0, mask, lambda_phys=0.01, alpha=0.01, nx=20, nt=20, dx=0.05, dt=0.05):
        batch_size = v0.shape[0]
        
        # 1. Forward Pass to get representations
        h0_probs, h0_states = self.rbms[0].sample_hidden(v0)
        h1_probs, h1_states = self.rbms[1].sample_hidden(h0_states)
        
        # 2. Reconstruct (Backward Pass)
        h0_recon_prob, h0_recon_state = self.rbms[1].sample_visible(h1_states)
        v1_mean, v1_sample = self.rbms[0].sample_visible(h0_recon_state)
        
        # 3. Hidden expectation under reconstructed visible
        h0_neg_prob, _ = self.rbms[0].sample_hidden(v1_sample)
        h1_neg_prob, _ = self.rbms[1].sample_hidden(h0_neg_prob)
        
        # 4. Standard CD-1 Update on layers
        # RBM 1
        dW1 = (np.dot(v0.T, h0_probs) - np.dot(v1_sample.T, h0_neg_prob)) / (batch_size * self.sigma)
        dv_bias1 = np.mean(v0 - v1_sample, axis=0) / (self.sigma**2)
        dh_bias1 = np.mean(h0_probs - h0_neg_prob, axis=0)
        
        # RBM 2
        dW2 = (np.dot(h0_states.T, h1_probs) - np.dot(h0_recon_state.T, h1_neg_prob)) / batch_size
        dv_bias2 = np.mean(h0_states - h0_recon_state, axis=0)
        dh_bias2 = np.mean(h1_probs - h1_neg_prob, axis=0)
        
        # 5. Physics Gradient propagation
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
            
            # Gradients for RBM 1
            dW1_phys = self.sigma * np.dot(grad_v_phys.T, h0_probs) / batch_size
            dv_bias1_phys = np.mean(grad_v_phys, axis=0)
            
            grad_h0_phys = self.sigma * np.dot(grad_v_phys, self.rbms[0].W)
            # Sigmoid derivative at h0_recon_prob
            grad_z1_phys = grad_h0_phys * h0_recon_prob * (1 - h0_recon_prob)
            
            # Gradients for RBM 2
            dW2_phys = np.dot(grad_z1_phys.T, h1_probs) / batch_size
            dv_bias2_phys = np.mean(grad_z1_phys, axis=0)
            
            dW1 -= lambda_phys * dW1_phys
            dv_bias1 -= lambda_phys * dv_bias1_phys
            dW2 -= lambda_phys * dW2_phys
            dv_bias2 -= lambda_phys * dv_bias2_phys

        # Gradient clipping
        np.clip(dW1, -1, 1, out=dW1)
        np.clip(dv_bias1, -1, 1, out=dv_bias1)
        np.clip(dh_bias1, -1, 1, out=dh_bias1)
        np.clip(dW2, -1, 1, out=dW2)
        np.clip(dv_bias2, -1, 1, out=dv_bias2)
        np.clip(dh_bias2, -1, 1, out=dh_bias2)

        self.rbms[0].W += self.lr * dW1
        self.rbms[0].v_bias += self.lr * dv_bias1
        self.rbms[0].h_bias += self.lr * dh_bias1
        
        self.rbms[1].W += self.lr * dW2
        self.rbms[1].v_bias += self.lr * dv_bias2
        self.rbms[1].h_bias += self.lr * dh_bias2
        
        error = np.mean(((v0 - v1_sample) * mask)**2)
        return error
