import numpy as np
from scipy.signal import convolve2d

class GaussianCRBM:
    def __init__(self, input_shape, filter_shape, n_filters, lr=0.001, sigma=0.1, random_state=42):
        """
        Physics-Informed Gaussian Convolutional Restricted Boltzmann Machine (PI-GCRBM).
        Visible units are continuous (Gaussian) 2D fields.
        Hidden units are 2D binary (Bernoulli) feature maps.
        """
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.n_filters = n_filters
        self.lr = lr
        self.sigma = sigma
        
        rng = np.random.default_rng(random_state)
        # Weights: (n_filters, FH, FW)
        self.W = rng.normal(0.0, 0.01, (n_filters, *filter_shape))
        self.v_bias = 0.0
        self.h_biases = np.zeros(n_filters)
        
        self.output_shape = (input_shape[0] - filter_shape[0] + 1, 
                             input_shape[1] - filter_shape[1] + 1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sample_hidden(self, v):
        """
        v: (H, W) continuous visible grid
        """
        h_probs = []
        h_states = []
        for k in range(self.n_filters):
            # Convolution: valid mode to get hidden feature map
            conv_out = convolve2d(v, self.W[k][::-1, ::-1], mode='valid')
            prob = self._sigmoid(conv_out / self.sigma + self.h_biases[k])
            state = (prob > np.random.rand(*prob.shape)).astype(np.float32)
            h_probs.append(prob)
            h_states.append(state)
        return np.array(h_probs), np.array(h_states)

    def sample_visible(self, h_states):
        """
        h_states: (n_filters, OH, OW) hidden binary states
        """
        v_mean = np.zeros(self.input_shape)
        for k in range(self.n_filters):
            # Full convolution to restore original input dimension
            v_mean += convolve2d(h_states[k], self.W[k], mode='full')
        v_mean += self.v_bias
        v_sample = v_mean + np.random.randn(*self.input_shape) * self.sigma
        return v_mean, v_sample

    def train_step(self, v0, mask, lambda_phys=0.01, alpha=0.01, nx=20, nt=20, dx=0.05, dt=0.05):
        # Flattened v0 to 2D
        v0_2d = v0.reshape(self.input_shape)
        mask_2d = mask.reshape(self.input_shape)
        
        # Positive phase
        h0_probs, h0_states = self.sample_hidden(v0_2d)
        
        # Negative phase (CD-1)
        v1_mean, v1_sample = self.sample_visible(h0_states)
        h1_probs, h1_states = self.sample_hidden(v1_sample)
        
        # Calculate standard CD gradients
        dW = np.zeros_like(self.W)
        dh_biases = np.zeros_like(self.h_biases)
        dv_bias = np.mean(v0_2d - v1_sample) / (self.sigma**2)
        
        for k in range(self.n_filters):
            pos_grad = convolve2d(v0_2d, h0_probs[k][::-1, ::-1], mode='valid')
            neg_grad = convolve2d(v1_sample, h1_probs[k][::-1, ::-1], mode='valid')
            dW[k] = (pos_grad - neg_grad) / self.sigma
            dh_biases[k] = np.mean(h0_probs[k] - h1_probs[k])
            
        # Physics residual on v1_mean
        if lambda_phys > 0:
            V = v1_mean
            u_t = (V[1:-1, 1:] - V[1:-1, :-1]) / dt
            u_xx = (V[2:, :-1] - 2*V[1:-1, :-1] + V[:-2, :-1]) / (dx**2)
            res = u_t - alpha * u_xx
            
            grad_V = np.zeros_like(V)
            grad_V[1:-1, 1:] += 2 * res / dt
            grad_V[1:-1, :-1] -= 2 * res / dt
            grad_V[2:, :-1] -= 2 * res * alpha / (dx**2)
            grad_V[1:-1, :-1] += 2 * res * 2 * alpha / (dx**2)
            grad_V[:-2, :-1] -= 2 * res * alpha / (dx**2)
            
            # Gradients with respect to weights and biases
            dv_bias_phys = np.mean(grad_V)
            dW_phys = np.zeros_like(self.W)
            for k in range(self.n_filters):
                dW_phys[k] = convolve2d(grad_V, h0_probs[k][::-1, ::-1], mode='valid')
                
            dW -= lambda_phys * dW_phys
            dv_bias -= lambda_phys * dv_bias_phys
            
        # Apply Updates with gradient clipping
        np.clip(dW, -1, 1, out=dW)
        np.clip(dh_biases, -1, 1, out=dh_biases)
        dv_bias = np.clip(dv_bias, -1, 1)
        
        self.W += self.lr * dW
        self.h_biases += self.lr * dh_biases
        self.v_bias += self.lr * dv_bias
        
        error = np.mean(((v0_2d - v1_sample) * mask_2d)**2)
        return error

    def reconstruct(self, v_init, mask, steps=10):
        v = v_init.reshape(self.input_shape)
        mask_2d = mask.reshape(self.input_shape)
        
        for _ in range(steps):
            h_probs, h_states = self.sample_hidden(v)
            v_mean, v_sample = self.sample_visible(h_states)
            v = v * mask_2d + v_sample * (1 - mask_2d)
            
        return v_mean.flatten()
