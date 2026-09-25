import torch
import numpy as np

class VAEClassifierPipeline:
    """
    A wrapper class that behaves like a standard Scikit-Learn model.
    It takes raw features, passes them through a trained VAE to extract
    the compressed latent representations, and then feeds those representations
    into a downstream classifier (e.g., Gradient Boosting, Random Forest).
    """
    def __init__(self, vae_model, classifier_model):
        self.vae = vae_model
        self.classifier = classifier_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.vae.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            Z = self.vae.get_latent(x_t, deterministic=True).cpu().numpy()
        return self.classifier.predict(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.vae.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            Z = self.vae.get_latent(x_t, deterministic=True).cpu().numpy()
        
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(Z)
        else:
            raise AttributeError("Downstream classifier does not support predict_proba.")
