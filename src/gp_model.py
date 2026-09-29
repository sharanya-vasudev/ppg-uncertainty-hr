"""
Sparse variational Gaussian Process regression via GPyTorch (Phase 9).

Why "sparse variational": an exact GP needs an n x n kernel matrix and O(n^3)
time - infeasible for ~60,000 training windows per fold. SVGP summarizes the
training set with m learned inducing points (m << n) and trains on
mini-batches, so cost per step depends on batch size and m, not on n.

Predictive distribution of a NEW OBSERVATION y* at input x*:
    y* ~ Normal( mean(x*), var_latent(x*) + noise_var )
  var_latent : uncertainty about the underlying function (epistemic; grows
               when x* is unlike the training data)
  noise_var  : irreducible observation noise (aleatoric; one constant for all
               inputs in a standard GP)
A PREDICTION interval must include both terms.
"""
from __future__ import annotations

import numpy as np
import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset


class SVGPRegressor(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        n_inducing, n_dims = inducing_points.shape
        q = gpytorch.variational.CholeskyVariationalDistribution(n_inducing)
        strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, q, learn_inducing_locations=True
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_predict_svgp(X_train, y_train, X_test, n_inducing=500, n_epochs=40,
                     batch_size=1024, lr=0.05, seed=42, verbose=False):
    """Fit on the training set, predict for the test set. All standardization
    uses TRAINING statistics only. Returns values in original BPM units."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    y_mean, y_std = y_train.mean(), y_train.std()
    dtype = torch.float64  # float64 avoids Cholesky failures common in float32
    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=dtype)
    ytr = torch.tensor((y_train - y_mean) / y_std, dtype=dtype)
    Xte = torch.tensor((X_test - x_mean) / x_std, dtype=dtype)

    idx = rng.choice(len(Xtr), size=min(n_inducing, len(Xtr)), replace=False)
    model = SVGPRegressor(Xtr[idx].clone()).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(ytr))
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        losses.append(total / len(ytr))
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch + 1:>3}/{n_epochs}  loss {losses[-1]:.4f}")

    model.eval()
    likelihood.eval()
    means, latent_vars = [], []
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            latent = model(Xte[i:i + 1024])
            means.append(latent.mean)
            latent_vars.append(latent.variance)
    mean_z = torch.cat(means).numpy()
    latent_var_z = torch.cat(latent_vars).numpy()
    noise_var_z = float(likelihood.noise.item())

    return {
        "mean": mean_z * y_std + y_mean,
        "latent_std": np.sqrt(latent_var_z) * y_std,
        "noise_std": float(np.sqrt(noise_var_z) * y_std),
        "total_std": np.sqrt(latent_var_z + noise_var_z) * y_std,
        "losses": losses,
    }
