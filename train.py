import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
from absl import app, flags
from utils.configs import load_train_config, TrainConfig
from utils.logger import *
from modules.trainer import Trainer
from typing import Any, Callable, Optional
import os.path as osp

flags.DEFINE_string('gin_configs', None, 'gin config files.')

def train_loop(local_rank: int, world_size: int, config: TrainConfig):
    LOG_INFO("Loading Trainer....")
    trainer = Trainer(local_rank=local_rank, world_size = world_size, config=config)
    LOG_INFO("Complete to load Trainer!")
    LOG_INFO("Start Training.")
    trainer.train()
    LOG_INFO("End Training.")

def main(unused_argv):
    # configs.set_common_flags()
    config = load_train_config(save_config=True)
    add_file_handler(osp.join(config.checkpoint_path,"terminal_log.log") )
    world_size = torch.cuda.device_count()
    if world_size == 1:
        train_loop(0,world_size,config)
    else:
        LOG_INFO("Does not support multi-GPU yet.")

    return 


if __name__=='__main__':
    app.run(main)