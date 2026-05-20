import numpy as np

class GaussianClassicBM:
    def __init__(self, n_visible, n_hidden, lr=0.001, sigma=0.1, random_state=42):
        """
        Physics-Informed Gaussian Classic Boltzmann Machine (PI-GBM).
        A fully connected Boltzmann Machine where visible units are continuous (Gaussian)
        and hidden units are binary (Bernoulli).
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden
        self.lr = lr
        self.sigma = sigma
        
        rng = np.random.default_rng(random_state)
        # Initialize symmetric weight matrix with zero diagonal
        raw_W = rng.normal(0.0, 0.01, (self.n_total, self.n_total))
        self.W = (raw_W + raw_W.T) / 2.0
        np.fill_diagonal(self.W, 0.0)
        
        self.biases = np.zeros(self.n_total)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_step(self, state):
        """
        Runs one step of sequential Gibbs sampling over all units.
        Visible units are continuous, hidden units are binary.
        state: (batch_size, n_total)
        """
        batch_size = state.shape[0]
        new_state = state.copy()
        
        # Randomize update order
        indices = np.random.permutation(self.n_total)
        for i in indices:
            activation = np.dot(new_state, self.W[:, i]) + self.biases[i]
            if i < self.n_visible:
                # Gaussian unit: mean depends on other units
                # s_i ~ N(bias_i + sigma * sum(W_ij * s_j), sigma^2)
                # Note: W is already scaled or we can scale it here
                mean = self.biases[i] + self.sigma * activation
                new_state[:, i] = mean + np.random.randn(batch_size) * self.sigma
            else:
                # Bernoulli unit
                prob = self._sigmoid(activation / self.sigma)
                new_state[:, i] = (prob > np.random.rand(batch_size)).astype(np.float32)
                
        return new_state

    def run_gibbs_chain(self, initial_state, steps=10):
        state = initial_state.copy()
        for _ in range(steps):
            state = self.sample_step(state)
        return state

    def train_step(self, v0, mask, lambda_phys=0.01, alpha=0.01, nx=20, nt=20, dx=0.05, dt=0.05, gibbs_steps=10):
        batch_size = v0.shape[0]
        
        # 1. Positive Phase: Clamp visible units to data
        pos_state = np.zeros((batch_size, self.n_total), dtype=np.float32)
        pos_state[:, :self.n_visible] = v0
        # Initialize hidden units randomly
        pos_state[:, self.n_visible:] = (np.random.rand(batch_size, self.n_hidden) > 0.5).astype(np.float32)
        
        # Let hidden units settle under clamped visible units
        for _ in range(5):
            for i in range(self.n_visible, self.n_total):
                activation = np.dot(pos_state, self.W[:, i]) + self.biases[i]
                prob = self._sigmoid(activation / self.sigma)
                pos_state[:, i] = (prob > np.random.rand(batch_size)).astype(np.float32)
                
        pos_correlations = np.dot(pos_state.T, pos_state) / batch_size
        pos_biases = np.mean(pos_state, axis=0)

        # 2. Negative Phase: Free running (unclamped)
        neg_state = pos_state.copy()
        neg_state = self.run_gibbs_chain(neg_state, steps=gibbs_steps)
        
        neg_correlations = np.dot(neg_state.T, neg_state) / batch_size
        neg_biases = np.mean(neg_state, axis=0)

        # Gradient ascent update (contrastive divergence)
        dW = (pos_correlations - neg_correlations) / self.sigma
        db = (pos_biases - neg_biases) / (self.sigma**2)

        # 3. Physics Penalty on reconstruction
        # Calculate reconstruction mean for visible units: v_mean = bias_v + sigma * W_vh * h + sigma * W_vv * v
        v_mean = self.biases[:self.n_visible] + self.sigma * np.dot(pos_state, self.W[:, :self.n_visible])
        
        if lambda_phys > 0:
            V = v_mean.reshape(batch_size, nx, nt)
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
            
            grad_W_phys = np.dot(grad_v_phys.T, pos_state) / batch_size
            grad_W_total = np.zeros_like(self.W)
            grad_W_total[:self.n_visible, :] += grad_W_phys
            grad_W_total[:, :self.n_visible] += grad_W_phys.T
            # Make symmetric
            grad_W_total = (grad_W_total + grad_W_total.T) / 2.0
            grad_b_phys = np.mean(grad_v_phys, axis=0)
            
            dW -= lambda_phys * grad_W_total
            db[:self.n_visible] -= lambda_phys * grad_b_phys

        # Gradient clipping
        np.clip(dW, -1, 1, out=dW)
        np.clip(db, -1, 1, out=db)

        self.W += self.lr * dW
        np.fill_diagonal(self.W, 0.0)
        self.biases += self.lr * db
        
        error = np.mean(((v0 - neg_state[:, :self.n_visible]) * mask)**2)
        return error

    def reconstruct(self, v_init, mask, steps=10):
        batch_size = v_init.shape[0]
        state = np.zeros((batch_size, self.n_total), dtype=np.float32)
        state[:, :self.n_visible] = v_init
        state[:, self.n_visible:] = (np.random.rand(batch_size, self.n_hidden) > 0.5).astype(np.float32)
        
        for _ in range(steps):
            state = self.sample_step(state)
            # Re-clamp the known elements
            state[:, :self.n_visible] = v_init * mask + state[:, :self.n_visible] * (1 - mask)
            
        # Return visible reconstruction mean
        v_mean = self.biases[:self.n_visible] + self.sigma * np.dot(state, self.W[:, :self.n_visible])
        return v_mean
