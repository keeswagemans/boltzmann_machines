import numpy as np

class ClassicBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.05, random_state=42):
        """
        A Classic (Normal) Boltzmann Machine with a fully connected recurrent graph.
        Total units in the system: N = n_visible + n_hidden.
        
        Unlike an RBM, visible units connect to visible units, and hidden units 
        connect to hidden units.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden
        self.lr = learning_rate
        
        rng = np.random.default_rng(random_state)
        
        raw_W = rng.normal(loc=0.0, scale=0.01, size=(self.n_total, self.n_total))
        self.W = (raw_W + raw_W.T) / 2.0
        
        np.fill_diagonal(self.W, 0.0)
        
        self.biases = np.zeros(self.n_total)

    def _sigmoid(self, x):
        """Numerically stable sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def run_gibbs_chain(self, initial_state, steps=30):
        """
        Performs asynchronous Gibbs sampling to let the network settle 
        into thermal equilibrium.
        
        Units cannot be updated in parallel (like an RBM). They must be updated 
        one by one in a randomized order because each unit's activation depends 
        on its neighbors within the same layer.
        """
        state = initial_state.copy()
        batch_size = state.shape[0]
        
        for _ in range(steps):
            unit_indices = np.random.permutation(self.n_total)
            
            for i in unit_indices:
                activation = np.dot(state, self.W[:, i]) + self.biases[i]
                prob = self._sigmoid(activation)
                
                # Draw a Bernoulli sample 
                state[:, i] = (prob > np.random.rand(batch_size)).astype(np.float32)
                
        return state

    def train_step(self, v_data, gibbs_steps=40):
        """
        Executes a single iteration of Boltzmann learning (Gradient Ascent 
        on log-likelihood).
        
        Contains two distinct processing phases:
          1. Positive Phase: Visible units are clamped to empirical data.
          2. Negative Phase: Network runs completely unclamped (free-running).
        """
        batch_size = v_data.shape[0]
        
        # Positive phase 
        pos_state = np.zeros((batch_size, self.n_total), dtype=np.float32)
        pos_state[:, :self.n_visible] = v_data
        
        pos_state[:, self.n_visible:] = (np.random.rand(batch_size, self.n_hidden) > 0.5).astype(np.float32)
        
        for _ in range(15):
            for i in range(self.n_visible, self.n_total):
                act = np.dot(pos_state, self.W[:, i]) + self.biases[i]
                pos_state[:, i] = (self._sigmoid(act) > np.random.rand(batch_size)).astype(np.float32)
                
        pos_correlations = np.dot(pos_state.T, pos_state) / batch_size
        pos_biases = np.mean(pos_state, axis=0)

        # Negative phase 
        neg_state = (np.random.rand(batch_size, self.n_total) > 0.5).astype(np.float32)
        neg_state = self.run_gibbs_chain(neg_state, steps=gibbs_steps)
        
        neg_correlations = np.dot(neg_state.T, neg_state) / batch_size
        neg_biases = np.mean(neg_state, axis=0)

        # Update weight matrices and biases via gradient difference 
        self.W += self.lr * (pos_correlations - neg_correlations)
        
        np.fill_diagonal(self.W, 0.0)
        
        self.biases += self.lr * (pos_biases - neg_biases)
        
        # Measure Mean Squared Error reconstruction loss of the data phase vs model phase
        recon_v = neg_state[:, :self.n_visible]
        return np.mean((v_data - recon_v) ** 2)

    def generate_samples(self, num_samples=5, steps=100):
        """
        Allows the fully trained normal Boltzmann machine to dream up and generate 
        completely novel data samples from random initial states.
        """
        initial_state = (np.random.rand(num_samples, self.n_total) > 0.5).astype(np.float32)
        final_state = self.run_gibbs_chain(initial_state, steps=steps)
        
        # Slice and return only the visible node window outputs
        return final_state[:, :self.n_visible]