import torch
from torch import Tensor
from utils.configs import TrainConfig
from utils.logger import * 
from jaxtyping import Float
from nerf_components.raybatch import DataBatch
from typing import Optional, Callable, Dict, Tuple
import gin
from utils.cv_utils.hybrid_math import * 
from utils.cv_utils.hybrid_operations import * 
from modules.loss.mono_depth_loss import ScaleAndShiftInvariantLoss
import math

def compute_psnr(x:Float[Tensor,"num_samples dim"], y:Float[Tensor,"num_samples dim"], mask:Optional[Float[Tensor,"num_samples"]]=None) \
    -> np.ndarray:
    mse = convert_numpy(compute_mse(x,y,mask))
    return -10.0 * np.log10(mse)

def compute_ssim(x:Float[Tensor,"num_samples dim"], y:Float[Tensor,"num_samples dim"], 
                 mask:Optional[Float[Tensor,"num_samples"]]=None) \
    -> np.ndarray:
    
    if mask:
        if len(mask.shape) == 2 and mask.shape[-1] == 1: mask = mask.squeeze(1)
        x = x[mask,:]
        y = y[mask,:]
    
    x = convert_numpy(x)
    y = convert_numpy(y)
    # SSIM
    k1 = 0.01
    k2 = 0.03
    
    mu1 = x.mean()
    mu2 = y.mean()
    
    var1 = np.var(x)
    var2 = np.var(y) 
     
    cov = np.cov(x.flatten(), y.flatten())[0][1]
    
    c1 = k1 ** 2
    c2 = k2 ** 2
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    ssim = numerator / denominator
    
    return ssim
         
@gin.configurable
def compute_mse(x:Float[Tensor,"num_samples dim"], y:Float[Tensor,"num_samples dim"], mask:Optional[Float[Tensor,"num_samples"]]=None):
    if mask:
        if len(mask.shape) == 2 and mask.shape[-1] == 1: mask = mask.squeeze(1)
        x = x[mask,:]
        y = y[mask,:]
    return torch.mean((x-y)**2)

@gin.configurable
def compute_rmse(x:Float[Tensor,"num_samples dim"], y:Float[Tensor,"num_samples dim"], mask:Optional[Float[Tensor,"num_samples"]]=None):
    if mask:
        if len(mask.shape) == 2 and mask.shape[-1] == 1: mask = mask.squeeze(1)
        x = x[mask,:]
        y = y[mask,:]
    return torch.mean(torch.sqrt((x-y)**2))

@gin.configurable
def compute_cos_sim_loss(x:Float[Tensor,"num_samples 3"], y:Float[Tensor,"num_samples 3"], mask:Optional[Float[Tensor,"num_samples"]]=None):
    if mask:
        if len(mask.shape) == 2 and mask.shape[-1] == 1: mask = mask.squeeze(1)
        x = x[mask,:]
        y = y[mask,:]
    sim = torch.sum(x*y, dim=-1)
    return torch.mean(1.-sim)

def compute_cos(x: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None):
    x = x.reshape(-1,3)
    y = y.reshape(-1,3)
    if mask is not None:
        if len(x.shape) == len(mask.shape) + 1:
            x = x[mask,:]
            y = y[mask,:]
        else:
            x = x[mask]
            y = y[mask]
    cos = np.sum(x*y, axis=1)
    return cos

@gin.configurable
def compute_eikonal(gradient:Float[Tensor,"num_samples 3"]):
    grad_norm = gradient.norm(2,dim=-1)
    eikonal = (grad_norm - 1.)**2
    return eikonal.mean()

@gin.configurable
def compute_normal_smooth(grad1:Float[Tensor,"num_samples 3"], grad2:Float[Tensor,"num_samples 3"], mask:Optional[Float[Tensor,"num_samples"]]=None):
    if mask:
        if len(mask.shape) == 2 and mask.shape[-1] == 1: mask = mask.squeeze(1)
        grad1 = grad1[mask,:]
        grad2 = grad2[mask,:]
    norm1 = grad1 / (grad1.norm(2,1,True) + 1e-5)
    norm2 = grad2 / (grad2.norm(2,1,True) + 1e-5)
    return torch.norm(norm1 - norm2, dim = -1).mean()

# modified from https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
@gin.configurable
def mono_prior_loss(x:Float[Tensor,"num_samples 1"], y:Float[Tensor,"num_samples 1"],helper_loss:ScaleAndShiftInvariantLoss,mask:Float[Tensor,"num_samples 1"]=None):
    k = torch.numel(x) // 1024
    width = math.floor(math.sqrt(k))
    height = k // width
        
    if width * height == k:
        target_shape = (1,width*32,height*32)
    else:
        target_shape = (1,k*32,32)
    if mask is None:
        mask = torch.ones_like(x)
    return helper_loss(x.reshape(target_shape), (y*50 + 0.5).reshape(target_shape),mask.reshape(target_shape))
        
@gin.configurable
class TotalLoss:
    def __init__(self,config:TrainConfig,
                 rgb_loss_fn:Callable = compute_mse,
                 depth_loss_fn:Callable = None,
                 normal_loss_fn:Callable = None,
                 eiknoal_loss_fn:Callable = None,
                 normal_smooth_fn:Callable = None,
                 end_step:int = 0
                 ):
        
        self.rgb_loss_fn     :Callable = rgb_loss_fn
        self.depth_loss_fn   :Callable = depth_loss_fn
        self.normal_loss_fn  :Callable = normal_loss_fn
        self.eiknoal_loss_fn :Callable = eiknoal_loss_fn
        self.normal_smooth_fn:Callable = normal_smooth_fn
        self.mono_prior = False
        if (depth_loss_fn is not None) and (type(depth_loss_fn) == type(mono_prior_loss)):
            self.scale_shift_invariant_loss = ScaleAndShiftInvariantLoss(0.5,1)
            self.mono_prior = True
        self.loss_weight:Dict[str,float] = config.loss_weight
        self.end_step:int = end_step # for depth and normal loss decay

    
    def compute_total_loss(self,raw_output:Dict[str,Tensor],render_output:Dict[str,Tensor], batch:DataBatch, step:int) \
        -> Tuple[Tensor, Dict[str,float]]:
            loss_dict = {}
            stat_dict = {}
            # color loss
            pred_rgb = render_output["rgb"]
            total_loss = self.loss_weight["rgb"] * self.rgb_loss_fn(pred_rgb,batch.color)
            loss_dict["rgb"] = total_loss.item()
            stat_dict["PSNR"] = compute_psnr(pred_rgb,batch.color)
            stat_dict["SSIM"] = compute_ssim(pred_rgb,batch.color)
            # coarse color loss for Coarse Network in NeRF
            if  "coarse_rgb" in render_output and self.loss_weight["coarse_rgb"] > 0:
                pred_coarse_rgb = render_output["coarse_rgb"]
                coarse_rgb_loss = self.loss_weight["coarse_rgb"] * self.rgb_loss_fn(pred_coarse_rgb,batch.color)
                total_loss += coarse_rgb_loss
                loss_dict["coarse_rgb"] = coarse_rgb_loss.item()
            
            decay = math.exp(step / self.end_step * 10) if self.end_step > 0 else 1.
            
            # mask = ((raw_output['sdf'] > 0.).any(dim=-1) & (raw_output['sdf'] < 0.).any(dim=-1))[None, :, None]
            # depth loss
            if self.depth_loss_fn and self.loss_weight["depth"] > 0:
                pred_depth = render_output["depth"]
                if self.mono_prior:
                    depth_loss = decay * self.loss_weight["depth"] * self.depth_loss_fn(pred_depth,batch.depth,self.scale_shift_invariant_loss)
                else: depth_loss = decay * self.loss_weight["depth"] * self.depth_loss_fn(pred_depth,batch.depth)
                total_loss += depth_loss
                loss_dict["depth"] = depth_loss.item()
                stat_dict["DEPTH_RMSE"] = compute_rmse(pred_depth,batch.depth)
            # normal loss 
            if self.normal_loss_fn and self.loss_weight["normal"] > 0:
                pred_normal = render_output["normal"]
                normal_loss = decay * self.loss_weight["normal"] * self.normal_loss_fn(pred_normal,batch.normal)
                total_loss += normal_loss
                loss_dict["normal"] = normal_loss.item()
                stat_dict["DEG"] =  np.rad2deg(np.arccos(compute_cos(convert_numpy(pred_normal),convert_numpy(batch.normal)))).mean()  
            # eikonal loss
            if self.eiknoal_loss_fn and self.loss_weight["eikonal"] > 0:
                eikonal = raw_output["grad_theta"]
                eikonal_loss = self.loss_weight["eikonal"] * self.eiknoal_loss_fn(eikonal)
                total_loss += eikonal_loss
                eikonal_nb = raw_output["grad_theta_nb"]
                eikonal_nb_loss = self.loss_weight["eikonal"] * self.eiknoal_loss_fn(eikonal_nb)
                total_loss += eikonal_nb_loss
                loss_dict["eikonal"] = eikonal_loss.item() + eikonal_nb_loss.item()
            # normal smooth loss 
            if self.normal_smooth_fn and self.loss_weight["normal_smooth"] > 0:
                norm1, norm2 = raw_output["grad_theta"],raw_output["grad_theta_nb"]
                normal_smooth_loss = self.loss_weight["normal_smooth"] * self.normal_smooth_fn(norm1,norm2)
                total_loss += normal_smooth_loss
                loss_dict["normal_smooth"] = normal_smooth_loss.item()
            
            loss_dict["total_loss"] = total_loss.item()
            return total_loss, loss_dict, stat_dict
            
             