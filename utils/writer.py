from torch.utils.tensorboard import SummaryWriter
import numpy as np
from jaxtyping import Float, Int8
from typing import Any, Dict


class Writer:
    def __init__(self, log_dir:str):
        self.log_dir = log_dir
    
    def write_image(self, name:str,image: Int8[np.ndarray, "HWC"], step:int):
        pass
    def write_scalar(self, name:str, scalar:Any, step:int):
        pass
    def write_figure(self, name:str, figure:Any, step:int):
        pass
    def write_render_dict(self,frame_id:int, render_dict:Dict[str,np.ndarray], step):
        pass
        
        
class TensorBoardWriter(Writer):
    def __init__(self, log_dir:str):
        self.writer = SummaryWriter(log_dir)
    
    def write_image(self, name:str,image: Int8[np.ndarray, "HWC"], step:int):
        self.writer.add_image(name,image,step, dataformats="HWC")
    
    def write_scalar(self, name:str, scalar:Any, step:int):
        self.writer.add_scalar(name,scalar,step)
    
    def write_figure(self, name:str, figure:Any, step:int):
        self.writer.add_figure(name,figure,step)

    def write_render_dict(self,frame_id:int, render_dict:Dict[str,np.ndarray], step):
        for key in render_dict:
            name = f"{key}/fid{frame_id:04d}"
            self.write_image(name,render_dict[key], step)