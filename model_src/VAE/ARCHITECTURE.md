# Tabular Variational Autoencoder (VAE) Architecture

This document outlines the architecture and implementation details of the Variational Autoencoder (`TabularVAE`) defined in `vae_model.py`. The model is specifically designed to compress multi-dimensional clinical features (tabular data) into a regularized, low-dimensional latent manifold.

## 1. Network Architecture

The VAE is implemented as a PyTorch `nn.Module` using a symmetric Encoder-Decoder structure.

### Encoder
The encoder compresses the input features down to the desired latent dimension.
- **Backbone**: A multi-layer perceptron (MLP) with two hidden layers.
- **Dimensionality Progression**: `in_features` $\rightarrow$ `128` $\rightarrow$ `64`.
- **Activations & Regularization**: Each linear layer is followed by Batch Normalization (`BatchNorm1d`), a LeakyReLU activation (negative slope `0.2`), and Dropout (rate `0.1`).
- **Latent Projections**: The output of the backbone splits into two parallel linear layers:
  - `fc_mu`: Predicts the mean vector ($\mu$) of the latent Gaussian distribution.
  - `fc_logvar`: Predicts the log-variance ($\log\sigma^2$).
  - Both output vectors have a size of `latent_dim=8` (by default).

### Reparameterization Trick
To allow backpropagation through the stochastic sampling process, the `reparameterize` method is used.
- **During Training**: Samples a latent vector $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
- **During Inference/Evaluation**: It deterministically uses the mean vector ($z = \mu$).

### Decoder
The decoder attempts to reconstruct the original input features from the compressed latent representation $z$.
- **Backbone**: A mirrored MLP that reverses the dimensionality reduction.
- **Dimensionality Progression**: `latent_dim (8)` $\rightarrow$ `64` $\rightarrow$ `128` $\rightarrow$ `in_features`.
- **Activations & Regularization**: Uses the identical combination of Batch Normalization, LeakyReLU, and Dropout as the encoder.

---

## 2. Key Methods and Functional Components

The `TabularVAE` class includes several built-in methods to handle training, inference, and persistence.

- **`forward(x)`**: 
  Executes the full forward pass. It sequentially encodes the input, applies the reparameterization trick, decodes the latent vector, and returns the tuple: `(x_recon, mu, logvar)`.

- **`get_latent(x, deterministic=True)`**: 
  A dedicated utility for extracting the compressed feature embeddings for downstream tasks (e.g., passing them to classical ML classifiers like Random Forest or AdaBoost). By default, it operates deterministically and simply returns $\mu$.

- **`loss_function(x_recon, x, mu, logvar, beta=0.005)`**: 
  Computes the **$\beta$-VAE loss**, which balances reconstruction quality and latent space regularization:
  - **Reconstruction Loss**: Computed as the Mean Squared Error (MSE) between the input `x` and the reconstruction `x_recon`.
  - **KL Divergence**: Acts as a regularizer, forcing the latent distributions to approximate a standard normal distribution ($\mathcal{N}(0, I)$).
  - **$\beta$ (Beta) Penalty**: A weighting scalar for the KL divergence (defaults to `0.005`). A higher $\beta$ encourages better feature disentanglement but may degrade raw reconstruction accuracy.

- **`save_model(filepath)` & `load_model(filepath, device)`**: 
  Custom helpers to save and load model checkpoints. These methods store both the network weights (`state_dict`) and essential architecture shape parameters (`in_features`, `latent_dim`) to ensure seamless model reloading across different environments.
