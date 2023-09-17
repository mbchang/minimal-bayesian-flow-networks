"""Implements the key equations in Section 4 of Bayesian Flow Networks."""

import torch


def compute_alpha_continuous(sigma_1, t):
    """Implements equation 74 in the paper."""
    alpha = -(2 * torch.log(sigma_1)) / (sigma_1 ** (2 * t))
    return alpha


def compute_alpha_discrete(sigma_1, i, n):
    """Implements equation 95 in the paper."""
    first_term = sigma_1 ** (-2 * (i / n))
    second_term = 1 - sigma_1 ** (2 / n)
    alpha = first_term * second_term
    return alpha


def compute_beta(sigma_1, t):
    """Implements equation 72 in the paper."""
    beta = sigma_1 ** (-2 * t) - 1
    return beta


def compute_gamma(sigma_1, t):
    """Implements equation 80 in the paper."""
    gamma = 1 - sigma_1 ** (2 * t)
    return gamma


def compute_entropy(beta, D):
    """Implements equation 65 in the paper."""
    x = torch.log(torch.tensor(2 * torch.pi * torch.e)) - torch.log(1 + beta)
    entropy = 0.5 * D * x
    return entropy


def bayesian_update_function(prev_theta, y, alpha):
    """Implements equations 49 and 50 in the paper."""
    prev_mu, prev_rho = prev_theta
    new_rho = prev_rho + alpha
    new_mu = (prev_mu * prev_rho + y * alpha) / new_rho
    return new_mu, new_rho


def bayesian_update_distribuion(prev_theta, x, alpha):
    """Implements equation 52 in the paper."""
    new_mu, new_rho = bayesian_update_function(prev_theta, x, alpha)
    new_var = alpha / (new_rho**2)
    return new_mu, new_var


def bayesian_flow_distribution(x, t, sigma_1):
    """Implements equation 77 in the paper."""
    gamma = compute_gamma(sigma_1, t)
    mu = gamma * x
    var = gamma * (1 - gamma)
    return mu, var


def estimate_x(mu, gamma, noise_estimate):
    """Implements equation 84 in the paper."""
    first_term = mu / gamma
    second_term = ((1 - gamma) / (gamma)) ** 0.5 * noise_estimate
    return first_term - second_term


def predict_output(network, mu, t, gamma, x_min=-1, x_max=1):
    """Use network to predict data given parameters and timestep.

    This implements the else statement of CTS_OUPUT_PREDICTION in the paper.

    Note that this assumes that rejection sampling had happened outside of this
    function to make sure that t is not too small
    """
    noise_estimate = network(mu, t)
    x = estimate_x(mu, gamma, noise_estimate)
    x_clipped = torch.clip(x, min=x_min, max=x_max)
    return x_clipped


def predict_output_with_gamma(network, mu, gamma, x_min=-1, x_max=1):
    """Use network to predict data given parameters and gamma.

    Exactly the same as predict_output(), but feeds gamma into the networks
    instead of feeding in t.
    """
    noise_estimate = network(mu, gamma)
    x = estimate_x(mu, gamma, noise_estimate)
    x_clipped = torch.clip(x, min=x_min, max=x_max)
    return x_clipped


def discrete_time_loss_fn(x, x_hat, alpha, n):
    """Implements equation 96."""
    squared_distance = (x - x_hat).norm(p=2, dim=-1, keepdim=True) ** 2
    loss = n * (alpha / 2) * squared_distance
    return loss.mean()  # (B, 1) -> ()


def continuous_time_loss_fn(x, x_hat, alpha):
    """Implements equation 101.

    Equation 101 instantiates equation 41 for continuous variables.

    Note that the factor of 2 in the denominator of the continuous loss in
    equation 41 cancels out with the factor of 2 in the numerator of the
    definition of alpha in equation 74.
    """
    squared_distance = (x - x_hat).norm(p=2, dim=-1, keepdim=True) ** 2
    loss = alpha * squared_distance  # (B, 1) -> ()
    return loss.mean()
