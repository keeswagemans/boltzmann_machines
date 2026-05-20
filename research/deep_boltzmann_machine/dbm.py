import numpy as np

class DeepBoltzmannMachine:
    def __init__(self, n_visible, n_h1, n_h2, learning_rate=0.01, random_state=42):
        """
        A 2-Hidden Layer Deep Boltzmann Machine (DBM).
        
        Structure: V <-> H1 <-> H2
        
        Parameters:
        --
        n_visible : int
            Number of visible units.
        n_h1 : int
            Number of units in the first hidden layer.
        n_h2 : int
            Number of units in the second hidden layer.
        learning_rate : float
            Learning rate for gradient ascent.
        """
        self.n_v = n_visible
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.lr = learning_rate
        
        rng = np.random.default_rng(random_state)
        
        # Weight matrices
        self.W1 = rng.normal(0, 0.01, (n_visible, n_h1))
        self.W2 = rng.normal(0, 0.01, (n_h1, n_h2))
        
        # Biases
        self.v_bias = np.zeros(n_visible)
        self.h1_bias = np.zeros(n_h1)
        self.h2_bias = np.zeros(n_h2)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_h1(self, v, h2):
        """H1 depends on both V and H2."""
        activation = np.dot(v, self.W1) + np.dot(h2, self.W2.T) + self.h1_bias
        prob = self._sigmoid(activation)
        state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
        return prob, state

    def sample_v(self, h1):
        """V depends on H1."""
        activation = np.dot(h1, self.W1.T) + self.v_bias
        prob = self._sigmoid(activation)
        state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
        return prob, state

    def sample_h2(self, h1):
        """H2 depends on H1."""
        activation = np.dot(h1, self.W2) + self.h2_bias
        prob = self._sigmoid(activation)
        state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
        return prob, state

    def train_step(self, v_data, n_gibbs=5):
        """
        Perform one step of DBM training using a simplified mean-field 
        approximation for the positive phase and Gibbs sampling for the negative phase.
        """
        batch_size = v_data.shape[0]
        
        # --- Positive Phase (Mean Field Approximation) ---
        # Initialize hidden layers randomly or with a small value
        mu1 = np.full((batch_size, self.n_h1), 0.5)
        mu2 = np.full((batch_size, self.n_h2), 0.5)
        
        # Fixed-point iteration to find mean-field parameters
        for _ in range(10):
            mu1 = self._sigmoid(np.dot(v_data, self.W1) + np.dot(mu2, self.W2.T) + self.h1_bias)
            mu2 = self._sigmoid(np.dot(mu1, self.W2) + self.h2_bias)
            
        # --- Negative Phase (Stochastic Approximation / Gibbs Sampling) ---
        # In a real DBM, we'd use persistent chains (PCD)
        v_neg = v_data.copy()
        h1_neg = (np.random.rand(batch_size, self.n_h1) > 0.5).astype(np.float32)
        h2_neg = (np.random.rand(batch_size, self.n_h2) > 0.5).astype(np.float32)
        
        for _ in range(n_gibbs):
            _, h1_neg = self.sample_h1(v_neg, h2_neg)
            _, v_neg = self.sample_v(h1_neg)
            _, h2_neg = self.sample_h2(h1_neg)
            
        # --- Update Weights and Biases ---
        # Gradient for W1: <v*h1>_pos - <v*h1>_neg
        pos_v_h1 = np.dot(v_data.T, mu1) / batch_size
        neg_v_h1 = np.dot(v_neg.T, h1_neg) / batch_size
        self.W1 += self.lr * (pos_v_h1 - neg_v_h1)
        
        # Gradient for W2: <h1*h2>_pos - <h1*h2>_neg
        pos_h1_h2 = np.dot(mu1.T, mu2) / batch_size
        neg_h1_h2 = np.dot(h1_neg.T, h2_neg) / batch_size
        self.W2 += self.lr * (pos_h1_h2 - neg_h1_h2)
        
        # Biases
        self.v_bias += self.lr * np.mean(v_data - v_neg, axis=0)
        self.h1_bias += self.lr * np.mean(mu1 - h1_neg, axis=0)
        self.h2_bias += self.lr * np.mean(mu2 - h2_neg, axis=0)
        
        return np.mean((v_data - v_neg)**2)

if __name__ == "__main__":
    dbm = DeepBoltzmannMachine(n_visible=10, n_h1=5, n_h2=2)
    data = np.random.rand(100, 10)
    for i in range(10):
        error = dbm.train_step(data)
        print(f"Step {i}, Reconstruction Error: {error:.4f}")
