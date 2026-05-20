import numpy as np
import sys
import os

# Add the restricted_boltzmann directory to the path so we can reuse the RBM class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'restricted_boltzmann')))
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, random_state=42):
        """
        Deep Belief Network (DBN) - A stack of Restricted Boltzmann Machines.
        
        Parameters:
        layer_sizes : list of int
            List containing the number of units in each layer.
            e.g., [784, 500, 200] where 784 is the input (visible) layer.
        learning_rate : float
            Learning rate for training.
        random_state : int
            Seed for reproducibility.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.rbms = []
        
        for i in range(len(layer_sizes) - 1):
            rbm = RestrictedBoltzmannMachine(
                n_visible=layer_sizes[i],
                n_hidden=layer_sizes[i+1],
                learning_rate=learning_rate,
                random_state=random_state + i
            )
            self.rbms.append(rbm)

    def pretrain(self, X, epochs=10, batch_size=64, k=1):
        """
        Greedy layer-wise unsupervised pre-training.
        """
        current_input = X
        
        for i, rbm in enumerate(self.rbms):
            print(f"Pre-training Layer {i+1}: {rbm.n_visible} -> {rbm.n_hidden}")
            
            for epoch in range(epochs):
                indices = np.random.permutation(len(current_input))
                total_error = 0
                
                for start_idx in range(0, len(current_input), batch_size):
                    batch = current_input[indices[start_idx : start_idx + batch_size]]
                    error = rbm.contrastive_divergence(batch, k=k)
                    total_error += error
                
                print(f"  Epoch {epoch+1}/{epochs}, Mean Error: {total_error / (max(1, len(current_input) // batch_size)):.4f}")
            
            # Pass the hidden probabilities of the current RBM as input to the next layer
            _, current_input = rbm.sample_hidden(current_input)

    def transform(self, X):
        """
        Passes input through the DBN to get the highest level latent representation.
        """
        current_input = X
        for rbm in self.rbms:
            _, current_input = rbm.sample_hidden(current_input)
        return current_input

    def reconstruct(self, X):
        """
        Reconstructions by passing up and down.
        """
        # Go up
        h_states = [X]
        for rbm in self.rbms:
            _, next_h = rbm.sample_hidden(h_states[-1])
            h_states.append(next_h)
        
        # Go down
        current_recon = h_states[-1]
        for rbm in reversed(self.rbms):
            _, current_recon = rbm.sample_visible(current_recon)
            
        return current_recon

if __name__ == "__main__":
    # Quick test with random data
    dbn = DeepBeliefNetwork(layer_sizes=[10, 5, 2], learning_rate=0.1)
    data = np.random.rand(100, 10)
    dbn.pretrain(data, epochs=5, batch_size=10)
    latent = dbn.transform(data)
    print("Latent representation shape:", latent.shape)
    recon = dbn.reconstruct(data)
    print("Reconstruction shape:", recon.shape)
