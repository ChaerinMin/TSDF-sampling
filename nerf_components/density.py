import torch.nn as nn
import torch
import gin
import numpy as np

@gin.configurable
class Density(nn.Module):
    def __init__(self, params_init={}):
        super(Density,self).__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

@gin.configurable
class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={"beta":0.1}, beta_min=0.0001):
        super(LaplaceDensity,self).__init__(params_init=params_init)
        beta_min = torch.tensor(beta_min)
        # self.beta = nn.Parameter(torch.tensor(params_init["beta"]))
        self.register_buffer("beta_min", beta_min)

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)

# for neus
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, shape):
        return torch.exp(self.variance * 10.0).reshape(1,1).repeat(shape)