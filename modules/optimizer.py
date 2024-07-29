import torch
import torch.optim as optim
from torch.optim import Optimizer
from utils.configs import TrainConfig


def get_optimizer(config:TrainConfig, model):
    if config.optimizer == "Adam":
        ## for debug
        # optimizer = optim.Adam([
        #     {'name': 'grid', 'params': list(model.network.geometric_field.encoder.grid_parameter()),\
        #         'lr': config.lr * config.multiplier_grid_lr},
        #     {'name': 'mlp', 'params': model.get_mlp_params() , 'lr': config.lr},
        #     {'name': 'density', 'params': model.network.geometric_field.density_fn.parameters(),\
        #         'lr': config.lr},
        #     ], betas=config.betas, eps=1e-15)
        optimizer = optim.Adam(model.parameters(), lr = config.lr, betas=config.betas, eps=1.0e-15)    
    return optimizer

def get_scheduler(config:TrainConfig, optimizer:Optimizer, decay_steps:int):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, config.decay_rate ** (1./decay_steps))