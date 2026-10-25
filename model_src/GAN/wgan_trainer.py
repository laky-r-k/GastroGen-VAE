"""
WGAN-GP Model Trainer Module.
Pure PyTorch training engine handling Wasserstein loss, gradient penalty,
and step-based minibatch optimization without embedded evaluation logic.
"""

import torch
import torch.autograd as autograd
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import TensorDataset, DataLoader

from .wgan_gp_model import Generator, Critic


class WGANGPTrainer:
    """
    Manages WGAN-GP training lifecycle, step-based training loop, and sampling.
    Strictly handles PyTorch optimization with no evaluation bloat.
    """

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        betas: tuple = (0.5, 0.9),
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        device: str = "cpu",
    ):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=betas
        )
        self.opt_d = torch.optim.Adam(
            self.critic.parameters(), lr=lr_d, betas=betas
        )

        self.history = {
            "step": [],
            "d_loss": [],
            "g_loss": [],
            "gp": [],
            "wasserstein_estimate": [],
        }

    def compute_gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        labels_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates 1-Lipschitz gradient penalty on randomly interpolated points.
        Interpolates per-sample with the label passed through.
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolates = (alpha * real_samples + (1.0 - alpha) * fake_samples).requires_grad_(True)

        d_interpolates = self.critic(interpolates, labels_onehot)

        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
        return gradient_penalty

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        generator_steps: int = 5000,
        batch_size: int = 64,
        checkpoint_interval: int = 500,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Executes step-based WGAN-GP training loop.
        Random minibatches are sampled each iteration to overcome small-sample constraints.
        """
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        num_classes = self.generator.num_classes

        # One-hot encode labels
        labels_onehot = torch.zeros(len(y_train), num_classes, dtype=torch.float32)
        labels_onehot.scatter_(1, y_tensor.unsqueeze(1), 1.0)

        dataset = TensorDataset(x_tensor, labels_onehot, y_tensor)
        n_samples = len(dataset)

        self.generator.train()
        self.critic.train()

        g_step = 0
        while g_step < generator_steps:
            # ---------------------
            #  Train Critic (n_critic steps)
            # ---------------------
            d_loss_val = 0.0
            gp_val = 0.0
            w_dist_val = 0.0

            for _ in range(self.n_critic):
                # Random minibatch sampling
                idx = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
                real_x = x_tensor[idx].to(self.device)
                real_oh = labels_onehot[idx].to(self.device)
                cur_batch_size = real_x.size(0)

                # Sample noise and fake samples
                z = torch.randn(cur_batch_size, self.generator.latent_dim, device=self.device)
                fake_x = self.generator(z, real_oh)

                # Critic forward pass
                d_real = self.critic(real_x, real_oh)
                d_fake = self.critic(fake_x.detach(), real_oh)

                # Gradient penalty
                gp = self.compute_gradient_penalty(real_x, fake_x.detach(), real_oh)

                # WGAN Critic Loss: maximize E[D(real)] - E[D(fake)] => minimize E[D(fake)] - E[D(real)]
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + self.lambda_gp * gp

                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()

                d_loss_val += d_loss.item()
                gp_val += gp.item()
                # Empirical Wasserstein distance estimate = E[D(real)] - E[D(fake)]
                w_dist_val += (torch.mean(d_real) - torch.mean(d_fake)).item()

            d_loss_val /= self.n_critic
            gp_val /= self.n_critic
            w_dist_val /= self.n_critic

            # ---------------------
            #  Train Generator (1 step)
            # ---------------------
            idx = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            gen_labels_oh = labels_onehot[idx].to(self.device)
            cur_batch_size = gen_labels_oh.size(0)

            z = torch.randn(cur_batch_size, self.generator.latent_dim, device=self.device)
            fake_x = self.generator(z, gen_labels_oh)

            # Generator objective: maximize E[D(fake)] => minimize -E[D(fake)]
            g_loss = -torch.mean(self.critic(fake_x, gen_labels_oh))

            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()

            g_step += 1

            # Log histories
            self.history["step"].append(g_step)
            self.history["d_loss"].append(d_loss_val)
            self.history["g_loss"].append(g_loss.item())
            self.history["gp"].append(gp_val)
            self.history["wasserstein_estimate"].append(w_dist_val)

            if verbose and (g_step % checkpoint_interval == 0 or g_step == generator_steps):
                print(
                    f"  [WGAN-GP] Step {g_step:5d}/{generator_steps} | "
                    f"D Loss: {d_loss_val:8.4f} | G Loss: {g_loss.item():8.4f} | "
                    f"W-Dist: {w_dist_val:8.4f} | GP: {gp_val:6.4f}"
                )

        return self.history

    def generate(self, n_per_class: Dict[int, int]) -> Dict[int, np.ndarray]:
        """
        Generates synthetic scaled feature arrays separated by class.
        Returns:
            Dictionary mapping class_id -> np.ndarray of shape (n_samples, feature_dim)
        """
        self.generator.eval()
        samples_per_class = {}

        with torch.no_grad():
            for cls_idx, count in n_per_class.items():
                if count <= 0:
                    samples_per_class[cls_idx] = np.empty((0, self.generator.output_dim))
                    continue

                # Batch generation to avoid GPU/CPU memory spikes if large
                batch_size = 256
                generated_chunks = []
                remaining = count

                while remaining > 0:
                    cur_n = min(batch_size, remaining)
                    z = torch.randn(cur_n, self.generator.latent_dim, device=self.device)
                    # One-hot label vector for this specific class
                    oh = torch.zeros(cur_n, self.generator.num_classes, device=self.device)
                    oh[:, cls_idx] = 1.0

                    chunk = self.generator(z, oh).cpu().numpy()
                    generated_chunks.append(chunk)
                    remaining -= cur_n

                samples_per_class[cls_idx] = np.vstack(generated_chunks)

        return samples_per_class
