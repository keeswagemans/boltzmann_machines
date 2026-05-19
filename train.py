import numpy as np
import matplotlib.pyplot as plt
from rbm import RestrictedBoltzmannMachine

def generate_mock_data(num_samples=1500):
    """Generates simple 8x8 blocks/bars images as structural patterns."""
    np.random.seed(42)
    data = np.zeros((num_samples, 64))
    for i in range(num_samples):
        img = np.zeros((8, 8))
        if np.random.rand() > 0.5:
            img[3:5, :] = 1.0  # Horizontal thick line
        else:
            img[:, 3:5] = 1.0  # Vertical thick line
        noise = np.random.rand(8, 8) < 0.04
        img = np.logical_xor(img, noise).astype(np.float32)
        data[i] = img.flatten()
    return data

def main():
    visible_units = 64  
    hidden_units = 16   
    epochs = 40
    batch_size = 32
    learning_rate = 0.25
    
    print("Generating structural dataset using uv runtime environment...")
    X = generate_mock_data()
    rbm = RestrictedBoltzmannMachine(n_visible=visible_units, n_hidden=hidden_units, learning_rate=learning_rate)
    
    print("\n--- Starting RBM Parameter Updates ---")
    for epoch in range(epochs):
        np.random.shuffle(X)
        epoch_errors = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            error = rbm.contrastive_divergence(batch, k=1)
            epoch_errors.append(error)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Reconstruction Error (MSE): {np.mean(epoch_errors):.5f}")

    # Display Reconstruction Results
    test_data = generate_mock_data(num_samples=4)
    reconstructed_data = rbm.reconstruct(test_data)
    
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(4):
        axes[0, i].imshow(test_data[i].reshape(8, 8), cmap='gray')
        axes[0, i].set_title(f"Orig {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed_data[i].reshape(8, 8), cmap='gray')
        axes[1, i].set_title(f"Recon {i+1}")
        axes[1, i].axis('off')
        
    plt.suptitle("Restricted Boltzmann Machine Reconstruction Performance")
    plt.tight_layout()
    print("\nDisplaying plot windows...")
    plt.show()

if __name__ == "__main__":
    main()