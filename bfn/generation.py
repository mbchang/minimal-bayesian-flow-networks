"""Implements sampling algorithms for Bayesian Flow Networks"""

import einops as eo
from collections import namedtuple
import torch

from bfn.config import SamplingConfig
from bfn.bayesian_flow_networks import (
    predict_output,
    compute_alpha_discrete,
    bayesian_update_function,
)

SamplingOutputs = namedtuple(
    "SamplingOutputs",
    ["samples", "idxs", "ts", "gammas", "alphas", "ys", "mus", "rhos"],
)


def simulate_trajectory(theta_0, alphas, x):
    thetas = [theta_0]
    ys = []

    for alpha in alphas:
        # sender distribution
        y = torch.normal(mean=x, std=1 / alpha**0.5)

        # bayesian update
        mu_i, rho_i = bayesian_update_function(prev_theta=thetas[-1], y=y, alpha=alpha)

        thetas.append((mu_i, rho_i))
        ys.append(y)

    mus, rhos = zip(*thetas)
    return mus, rhos, ys


def generate(network, indim, num_iters, num_samples, config=SamplingConfig()):
    """Implements Algorithm 3 in the paper."""
    with torch.no_grad():
        network.eval()

        # sampling hyperparameters
        sigma_1 = config.sigma_1
        t_min = config.t_min
        n = num_iters
        k = num_samples

        outputs = SamplingOutputs(
            samples=torch.zeros((k, n + 1, indim)),
            idxs=torch.zeros((k, n, 1), dtype=torch.long),
            ts=torch.zeros((k, n, 1)),
            gammas=torch.zeros((k, n, 1)),
            alphas=torch.zeros((k, n, 1)),
            ys=torch.zeros((k, n, indim)),
            mus=torch.zeros((k, n + 1, indim)),
            rhos=torch.zeros((k, n + 1, indim)),
        )

        # initialize
        outputs.mus[:, 0] = 0.0
        outputs.rhos[:, 0] = 1.0

        for idx in torch.arange(1, n + 1):
            i = eo.repeat(idx, "-> k 1", k=k)
            mu = outputs.mus[:, idx - 1]
            rho = outputs.rhos[:, idx - 1]

            # accuracy schedule
            t = (i - 1) / n
            gamma = 1 - sigma_1 ** (2 * t)
            alpha = compute_alpha_discrete(sigma_1, i, n)

            # output distribution
            x_hat = torch.where(
                condition=t < t_min,
                input=torch.zeros((k, indim)),
                other=predict_output(network, mu, t, gamma),
            )

            # receiver distribution
            y = torch.normal(mean=x_hat, std=alpha ** (-0.5))

            # bayesian update
            mu, rho = bayesian_update_function(prev_theta=(mu, rho), y=y, alpha=alpha)

            outputs.samples[:, idx - 1] = x_hat
            outputs.idxs[:, idx - 1] = i
            outputs.ts[:, idx - 1] = t
            outputs.gammas[:, idx - 1] = gamma
            outputs.alphas[:, idx - 1] = alpha
            outputs.ys[:, idx - 1] = y
            outputs.mus[:, idx] = mu
            outputs.rhos[:, idx] = rho

        x_hat = predict_output(network, mu, torch.ones((k, 1)), 1 - sigma_1**2)

        outputs.samples[:, n] = x_hat

    return outputs
