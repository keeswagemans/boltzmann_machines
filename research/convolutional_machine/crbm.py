import numpy as np
from scipy.signal import convolve2d

class ConvolutionalRBM:
    def __init__(self, input_shape, filter_shape, n_filters, learning_rate=0.01, random_state=42):
        """
        Convolutional Restricted Boltzmann Machine (CRBM).
        
        Parameters:
        --
        input_shape : tuple (H, W)
            Shape of the input 2D image.
        filter_shape : tuple (FH, FW)
            Shape of the convolutional filters.
        n_filters : int
            Number of filters (feature maps).
        learning_rate : float
            Learning rate.
        """
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.n_filters = n_filters
        self.lr = learning_rate
        
        rng = np.random.default_rng(random_state)
        
        # Weights: (n_filters, FH, FW)
        self.W = rng.normal(0, 0.01, (n_filters, *filter_shape))
        
        # Biases
        self.v_bias = 0.0
        self.h_biases = np.zeros(n_filters)
        
        # Output shape of the hidden layer (valid convolution)
        self.output_shape = (input_shape[0] - filter_shape[0] + 1, 
                            input_shape[1] - filter_shape[1] + 1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_hidden(self, v):
        """
        v: (H, W) or (batch, H, W)
        Currently implemented for a single sample (H, W) for simplicity.
        """
        h_probs = []
        h_states = []
        
        for k in range(self.n_filters):
            # Convolution: V * W_k
            # We use 'valid' to match standard CRBM behavior
            conv_out = convolve2d(v, self.W[k][::-1, ::-1], mode='valid')
            prob = self._sigmoid(conv_out + self.h_biases[k])
            state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
            h_probs.append(prob)
            h_states.append(state)
            
        return np.array(h_probs), np.array(h_states)

    def sample_visible(self, h_states):
        """
        h_states: (n_filters, OH, OW)
        Reconstruct visible from hidden maps.
        """
        # Sum of (H_k * mirrored(W_k))
        v_act = np.zeros(self.input_shape)
        for k in range(self.n_filters):
            # 'full' convolution of H_k with W_k yields the original input size
            v_act += convolve2d(h_states[k], self.W[k], mode='full')
            
        prob = self._sigmoid(v_act + self.v_bias)
        state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
        return prob, state

    def train_step(self, v_data):
        """
        A single CD-1 training step for the CRBM.
        v_data: (H, W)
        """
        # Positive phase
        h0_probs, h0_states = self.sample_hidden(v_data)
        
        # Negative phase (CD-1)
        v1_probs, v1_states = self.sample_visible(h0_states)
        h1_probs, h1_states = self.sample_hidden(v1_states)
        
        # Gradient updates
        for k in range(self.n_filters):
            # dW_k = conv(V_pos, H_pos_k) - conv(V_neg, H_neg_k)
            # Need to be careful with convolution modes and orientations
            # For dW, it's essentially a correlation
            pos_grad = convolve2d(v_data, h0_probs[k][::-1, ::-1], mode='valid')
            neg_grad = convolve2d(v1_states, h1_probs[k][::-1, ::-1], mode='valid')
            
            self.W[k] += self.lr * (pos_grad - neg_grad)
            self.h_biases[k] += self.lr * np.mean(h0_probs[k] - h1_probs[k])
            
        self.v_bias += self.lr * np.mean(v_data - v1_states)
        
        return np.mean((v_data - v1_states)**2)

if __name__ == "__main__":
    # Test with a small image
    crbm = ConvolutionalRBM(input_shape=(10, 10), filter_shape=(3, 3), n_filters=2)
    dummy_data = np.random.rand(10, 10)
    for i in range(10):
        error = crbm.train_step(dummy_data)
        print(f"Step {i}, Reconstruction Error: {error:.4f}")
