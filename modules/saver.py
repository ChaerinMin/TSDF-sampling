import torch
from nerf_components.network import *
from nerf_components.sampler import *
from modules.optimizer import *
import os

def save_checkpoint(
        path:str,
        step:int,
        network:BaseNetwork,
        sampler:Sampler,
        optimizer:Optimizer
):
    """
    path:
    step:int
    network:Network
    sampler
    optimizer
    scheduler
    """
    checkpoint_dict = {
        "network": network.state_dict(),
        "sampler": sampler.state_dict() if sampler is not None else None,
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(checkpoint_dict,path)

def load_checkpoint(path:str, map_location:Any=torch.device("cpu")) -> Dict[str,Any]:

    if os.path.exists(path) is False:
        LOG_ERROR(f"path:{path} does not exist.")
        return None
    checkpoint_dict:Dict[str,Any] = torch.load(path,map_location)
    return checkpoint_dict 



