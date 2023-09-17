"""Implements training algorithms for Bayesian Flow Networks"""

from tenacity import retry
import torch
import torch.optim as optim

from bfn.config import TrainingConfig, SamplingConfig
from bfn.bayesian_flow_networks import (
    predict_output,
    compute_alpha_continuous,
    continuous_time_loss_fn,
)


@retry
def sample_t(t_min):
    t = torch.rand(size=())
    if t < t_min:
        print(f"Sampled t = {t} thus t < {t_min}. Retrying!")
        raise Exception
    else:
        return t


def train(network, x, train_cfg=TrainingConfig(), generate_cfg=SamplingConfig()):
    sigma_1 = torch.tensor(generate_cfg.sigma_1)

    optimizer = optim.Adam(network.parameters(), lr=train_cfg.lr)

    ts = torch.zeros(train_cfg.steps)
    gammas = torch.zeros(train_cfg.steps)
    mus = torch.zeros((train_cfg.steps, *x.shape))
    alphas = torch.zeros(train_cfg.steps)
    losses = torch.zeros(train_cfg.steps // train_cfg.batch_size)

    for step in range(train_cfg.steps):
        t = sample_t(generate_cfg.t_min)  # guarantee t >= t_min
        gamma = 1 - sigma_1 ** (2 * t)
        mu = torch.normal(gamma * x, (gamma * (1 - gamma)) ** 0.5)

        ts[step] = t
        gammas[step] = gamma
        mus[step] = mu

        if step % train_cfg.batch_size == train_cfg.batch_size - 1:
            # construct batch
            batch_start = step - train_cfg.batch_size + 1
            batch_end = step + 1

            batch_mu = mus[batch_start:batch_end]
            batch_t = ts[batch_start:batch_end].unsqueeze(1)
            batch_gamma = gammas[batch_start:batch_end].unsqueeze(1)
            batch_x = x.repeat(train_cfg.batch_size, 1)

            # forward
            x_hat = predict_output(network, batch_mu, batch_t, batch_gamma)
            alpha = compute_alpha_continuous(sigma_1, batch_t)
            loss = continuous_time_loss_fn(batch_x, x_hat, alpha)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            alphas[batch_start:batch_end] = alpha.squeeze()
            losses[step // train_cfg.batch_size] = loss.item()

    return losses, ts, gammas, mus, alphas
