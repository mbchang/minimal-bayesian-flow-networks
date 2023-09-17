import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, indim, hdim):
        super().__init__()
        self.x_encoder = nn.Linear(indim, hdim)
        self.t_encoder = nn.Linear(1, hdim)
        self.decoder = nn.Linear(2 * hdim, indim)

    def forward(self, x, t):
        x_encoded = self.x_encoder(x)
        t_encoded = self.t_encoder(t)
        concatenated = torch.cat((x_encoded, t_encoded), dim=-1)
        concatenated = F.relu(concatenated)
        output = self.decoder(concatenated)
        return output
