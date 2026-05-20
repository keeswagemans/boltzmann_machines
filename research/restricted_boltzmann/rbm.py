import numpy as np

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, random_state=42):
        """
        Restricted Boltzmann Machine (RBM) using Contrastive Divergence (CD-k).
        
        Parameters:
        n_visible : int
            Number of visible units (input dimensions).
        n_hidden : int
            Number of hidden units (latent features).
        learning_rate : float
            Learning rate step size for gradient ascent.
        random_state : int
            Seed value to ensure deterministic weight initialization.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        
        # Initialize the weight matrix and biases
        # Weights are initialized from a small normal distribution to break symmetry
        rng = np.random.default_rng(random_state)
        self.W = rng.normal(loc=0.0, scale=0.01, size=(n_visible, n_hidden))
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)

    def _sigmoid(self, x):
        """Numerically stable sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_hidden(self, v):
        """
        Computes hidden layer activations given visible vector data.
        
        P(h_j = 1 | v) = sigmoid(b_j + sum_i v_i * W_ij)
        """
        activation = np.dot(v, self.W) + self.h_bias
        h_prob = self._sigmoid(activation)
        # Bernoulli trial sampling step
        h_state = (h_prob > np.random.rand(*h_prob.shape)).astype(np.float32)
        return h_prob, h_state

    def sample_visible(self, h):
        """
        Reconstructs visible layer activations given hidden activation states.
        
        P(v_i = 1 | h) = sigmoid(a_i + sum_j W_ij * h_j)
        """
        activation = np.dot(h, self.W.T) + self.v_bias
        v_prob = self._sigmoid(activation)
        # Bernoulli trial sampling step
        v_state = (v_prob > np.random.rand(*v_prob.shape)).astype(np.float32)
        return v_prob, v_state

    def contrastive_divergence(self, v0, k=1):
        """
        Performs a k-step Contrastive Divergence update on the RBM weights and biases.
        
        Parameters:
        v0 : ndarray of shape (batch_size, n_visible)
            The input batch representing empirical data configurations.
        k : int
            The number of Gibbs sampling steps to perform (default is CD-1).
            
        Returns:
        float
            The mean reconstruction error (MSE) across this mini-batch.
        """
        #  Positive Phase 
        # Calculate expectations using real training vector data samples
        h0_prob, h0_state = self.sample_hidden(v0)
        
        #  Negative Phase (Gibbs Sampling) 
        # Alternating updates to approximate the intractable partition function distribution
        vk = v0.copy()
        for _ in range(k):
            _, hk_state = self.sample_hidden(vk)
            _, vk = self.sample_visible(hk_state)
            
        # Final probability configuration evaluation
        hk_prob, _ = self.sample_hidden(vk)
        batch_size = v0.shape[0]
        
        #  Compute Vectorized Gradients & Apply Updates 
        # dW = (v0^T * h0) - (vk^T * hk)
        self.W += self.lr * (np.dot(v0.T, h0_prob) - np.dot(vk.T, hk_prob)) / batch_size
        
        # da = average error across visible units
        self.v_bias += self.lr * np.mean(v0 - vk, axis=0)
        
        # db = average error across hidden units
        self.h_bias += self.lr * np.mean(h0_prob - hk_prob, axis=0)
        
        # Evaluate performance using Mean Squared Error reconstruction loss
        error = np.mean((v0 - vk) ** 2)
        return error

    def reconstruct(self, v):
        """
        Helper method to infer hidden configurations from raw inputs and reconstruct 
        the data directly back onto the visible plane. Used for testing/evaluation.
        """
        _, h_state = self.sample_hidden(v)
        v_prob, _ = self.sample_visible(h_state)
        return v_prob