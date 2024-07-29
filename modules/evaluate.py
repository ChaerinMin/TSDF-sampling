
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from utils.configs import TrainConfig
from utils.logger import * 
from jaxtyping import Float
from nerf_components.raybatch import DataBatch
from typing import Optional, Callable, Dict,Union, Tuple
import gin
from utils.writer import *
from utils.cv_utils.hybrid_math import * 
from utils.cv_utils.hybrid_operations import * 
from utils.cv_utils.image_utils import * 
from modules.loss.losses import *
from models.model import *
from nerf_components.raybatch import *
import matplotlib.pyplot as plt
import cv2 as cv 
from matplotlib.patches import Circle


def figure_ray_distribution(pt2d:Tuple[int],
                          raydata:RayBatch,
                          ray_packet:Dict[str,Array],
                          render_images:Dict[str,Array],
                          vertical:bool=False,
                          important_ratio:float=0.99,
                          save_path:str=None,
                        ):
    """
    Plot Ray's Distribution
    
    Distributed values:
        1. weight distribution
        2. color distribution
        3. sdf distribution (optional)
    Vertical line:
        1. predected depth
        2. gt depth (optional)
        3. zero level for sdf (optional)
    extra info:
        1. num of importance samples     
    """
    fig = plt.figure(figsize=(20,10))
  
    if not "tsdf" in ray_packet:
        ax_img = fig.add_subplot(2,1,1) # show whole image
        ax_weight = fig.add_subplot(2,2,3) # show weight distribution
        # ax_acc = fig.add_subplot(2,2,4)
        # ax_weight_nb = fig.add_subplot(3,2,4) # show weight distribution close
    else:
        ax_img = fig.add_subplot(2,1,1)
        ax_weight = fig.add_subplot(2,2,3)
        ax_tsdf = fig.add_subplot(2,2,4)
    
    # show image & depth image
    h,w = render_images["rgb"].shape[0:2] 
    result = concat_images([render_images["rgb"],render_images["depth"],render_images["acc"]],vertical=vertical) # horizontal concat
    ax_img.imshow(result)
    ax_img.set_title(f"point(x,y) = ({pt2d[0]},{pt2d[1]})")
    ax_img.add_patch(Circle(pt2d, 5, color='red'))
    ax_img.add_patch(Circle((pt2d[0]+w, pt2d[1]), 5, color='red'))
    ax_img.add_patch(Circle((pt2d[0]+w*2,pt2d[1]), 5, color='red'))
    
    
    # show weight distribution
    weight = reduce_dim(convert_numpy(ray_packet['weight']),dim=0) 
    zs =  reduce_dim(convert_numpy(ray_packet['frustum'].zs) * raydata.scale,dim=0)  # denormalize
    num_sampels = zs.shape[-1]
    ax_weight.plot(zs, weight, 'k.-')
    acc = reduce_dim(convert_numpy(ray_packet["acc"]),dim=0)
    sorted_weight = np.sort(weight)[::-1]
    # compute how many sample contribute to render 
    score = 0.
    num_important_samples = 0
    while score < important_ratio and num_important_samples < weight.shape[0]:
        score += sorted_weight[num_important_samples].item()
        num_important_samples += 1
    # pred_color = convert_numpy(ray_packet["rgb"])
    
    # z_depth = reduce_dim(convert_numpy(ray_packet["z-depth"]),dim=0).item()  
    ray_depth = reduce_dim(convert_numpy(ray_packet["ray-depth"]),dim=0).item() 
    ax_weight.set_title(f"weigth distribution(sum of weight: {acc.item():.3f}, {important_ratio*100}% {num_important_samples}/{num_sampels})")
    ax_weight.axvline(ray_depth, color='r', label="pred ray-depth")
    # ax_weight.axvline(z_depth, color='b', label="pred z-depth")
    # ax_weight.legend(('weight', 'ray-depth:%.3f'%(ray_depth), 'z-depth:%.3f'%(z_depth)))    
    ax_weight.legend(('weight', 'ray-depth:%.3f'%(ray_depth)))    

    if "tsdf" in ray_packet:
        ax_tsdf.plot(zs, convert_numpy(ray_packet["tsdf"]), 'k.-')
        ax_tsdf.set_title(f"tsdf distribution")
        ax_tsdf.axhline(y=0., color='r')
    plt.subplots_adjust(hspace=0.5)
    if save_path is not None: plt.savefig(save_path)
    return fig

def figure_tsdf_distribution(pt2d:Tuple[int],
                          raydata:RayBatch,
                          ray_packet:Dict[str,Array],
                          render_images:Dict[str,Array],
                          vertical:bool=False):
    fig = plt.figure(figsize=(20,10))
  
    ax_img = fig.add_subplot(2,1,1) # show whole image
    ax_tsdf = fig.add_subplot(2,2,3) # show tsdf distribution

    h,w = render_images["rgb"].shape[0:2] 
    result = concat_images([render_images["rgb"],render_images["depth"],render_images["acc"]],vertical=vertical) # horizontal concat
    ax_img.imshow(result)
    ax_img.set_title(f"point(x,y) = ({pt2d[0]},{pt2d[1]})")
    ax_img.add_patch(Circle(pt2d, 5, color='red'))
    ax_img.add_patch(Circle((pt2d[0]+w, pt2d[1]), 5, color='red'))
    ax_img.add_patch(Circle((pt2d[0]+w*2,pt2d[1]), 5, color='red'))
    # show tsdf distribution
    tsdf = convert_numpy(ray_packet['tsdf']) 
    zs =  reduce_dim(convert_numpy(ray_packet['zs']),dim=0)  # denormalize
    ax_tsdf.plot(zs, tsdf, 'k.-')
    ax_tsdf.set_title(f"tsdf distribution")
    ax_tsdf.axhline(y=0., color='r')
        
    plt.subplots_adjust(hspace=0.5)
    # if save_path is not None: plt.savefig(save_path)
    return fig
    