from torch import Tensor
import torch.nn as nn
from nerf_components.encoding import *
from nerf_components.mlp import * 
from nerf_components.density import *
from nerf_components.field import * 
from nerf_components.raybatch import RayBatch 
import gin
from typing import Optional
from utils.logger import *
from utils.configs import TrainConfig
from nerf_components.network import *
from nerf_components.sampler import *
import time

@gin.configurable
def load_model(model_cls,network_fn,sampler_fn):
    network = network_fn()
    sampler = sampler_fn()
    model = model_cls(sampler,network)
    return model

@gin.configurable
class BaseModel(nn.Module):
    def __init__(self,
                sampler:Sampler,
                network:BaseNetwork                 
                   ):
        super(BaseModel,self).__init__()
        self.sampler = sampler
        self.network = network

    def forward(self, raydata:RayBatch, step:Optional[int]=None, sampler:Optional[Sampler]=None):
        if sampler is None: sampler = self.sampler
        input_dict,extra_output = sampler.sample(raydata, self.network, self.training)
        raw_output = self.network(input_dict)
        ## volume rendering
        render_output = self.sampler.volume_render(frustum=input_dict["frustum"],raw_output=raw_output)
        render_output["total_samples"] = raw_output["raw_rgbs"].shape[0]
        if "depth" in render_output:
            render_output["depth"] = render_output["depth"] * raydata.depth_scale * raydata.scale
        if len(extra_output) and self.training:
            if "coarse_rgb" in extra_output:
                render_output["coarse_rgb"] = extra_output["coarse_rgb"]
        if self.training:
            meta_data = {}
            return raw_output, render_output, meta_data
        else:
            return render_output
    
    def get_mlp_params(self):
        parameters = self.network.mlp_parameters()
        if isinstance(self.sampler, PdfSampler) and self.sampler.sample_network is not None:
            parameters += list(self.sampler.sample_network.mlp_parameters())
        return parameters
        
    def forward_sample_distribution(self,raydata:RayBatch):
        if raydata.batch_size != 1: 
            LOG_WARN("sample's raydata should be 1 ray.")
        input_dict,_ = self.sampler.sample(raydata, self.network, self.training)
        raw_output = self.network(input_dict)
        ## volume rendering
        render_output = self.sampler.volume_render(frustum=input_dict["frustum"],raw_output=raw_output,out_weight=True)
        if "depth" in render_output:
            render_output["z-depth"] = render_output["depth"] * raydata.depth_scale * raydata.scale
            render_output["ray-depth"] = render_output["depth"] * raydata.scale
        output_dict = {**raw_output, **render_output}
        output_dict["frustum"] = input_dict["frustum"]
        output_dict["position"] = input_dict["position"]
        return output_dict
        
    
@gin.configurable
class ModelWithTSDF(BaseModel):
    def __init__(self,
                sampler:TSDFSampler,
                network:BaseNetwork,
                step_update_begin:int=40000,
                step_use_tsdf:int=80000,
                step_per_update:int=1,
                margin_ratio:float = 0.0,
                use_resample:bool = True,
                acc_thr:float = 0.95,  # under this threshold, resample with the recovery algorithm
                ):
        super(ModelWithTSDF,self).__init__(sampler,network)
        self.step_update_begin = step_update_begin 
        self.step_use_tsdf = step_use_tsdf
        self.step_per_update = step_per_update
        self.margin_ratio = margin_ratio
        
        self.num_samples_divide_factor = 4
        self.num_fine_samples_divide_factor = 4
        self.acc_thr = acc_thr
        self.use_resample = use_resample
        
    def forward(self, raydata:RayBatch, step:int=None):
        # self.sampler.use_adaptive = True
        if self.training or self.sampler.use_tsdf is False:
            return super().forward(raydata,step,self.sampler.base_sampler)

        near_old, far_old = raydata.near[0].item(), raydata.far[0].item()
        num_samples, num_fine_samples = self.sampler.base_sampler.num_samples,self.sampler.base_sampler.num_fine_samples
        if self.sampler.use_adaptive is False:
            if self.sampler.use_tsdf:
                near,far = self.sampler.carving_empty_space(raydata,self.training)
                raydata.near = near * (1. - self.margin_ratio)
                raydata.far = far* (1. + self.margin_ratio)
                self.sampler.base_sampler.num_samples = num_samples // self.num_samples_divide_factor
                self.sampler.base_sampler.num_fine_samples = num_fine_samples // self.num_fine_samples_divide_factor
                render_output = super().forward(raydata,step)
            self.sampler.base_sampler.num_samples = num_samples
            self.sampler.base_sampler.num_fine_samples = num_fine_samples
        else:
            render_output = self._forward_adaptive(raydata) # use adaptive
        invalid_mask = (render_output["acc"] < self.acc_thr).squeeze()
        if self.use_resample and invalid_mask.sum():
            invalid_raydata = raydata[invalid_mask]
            invalid_raydata.near = torch.ones_like(invalid_raydata.near) * near_old 
            invalid_raydata.far = torch.ones_like(invalid_raydata.far) * far_old
            invalid_output = super().forward(invalid_raydata,step,self.sampler.base_sampler)
            for key in render_output:
                if "total_samples" == key:
                    render_output[key] += invalid_output[key]
                else:
                    render_output[key][invalid_mask] = invalid_output[key]
        render_output["under_acc"] = invalid_mask.sum().item()
        
        return render_output
    
    def _forward_adaptive(self, raydata:RayBatch):
        input_dict, _ = self.sampler.sample(raydata,self.network, self.training)
        raw_output = self.network(input_dict)
        render_output = self.sampler.volume_render_adaptive(input_dict["rays"],\
                            input_dict["deltas"],input_dict["zs"],raw_output=raw_output)
        if "depth" in render_output:
            render_output["depth"] = render_output["depth"] * raydata.depth_scale * raydata.scale
        return render_output
        
    def update_tsdf(self,raydata:RayBatch, depth:Optional[Tensor]=None):
            prev_state = self.sampler.use_tsdf 
            self.sampler.use_tsdf = False
            if depth is None:
                _, render_output = super().forward(raydata)
                depth = render_output["depth"]
            ray_depth = depth / (raydata.depth_scale * raydata.scale)
            origin,direction = raydata.get_origin_direction(normalize=True)
            self.sampler.update_tsdf_grid(origin,direction,ray_depth)
            self.sampler.use_tsdf = prev_state

    def forward_using_tsdf(self,raydata:RayBatch):
        prev_state = self.sampler.use_tsdf 
        self.sampler.use_tsdf = True
        raw_output, render_output = super().forward(raydata)
        self.sampler.use_tsdf = prev_state
        return raw_output, render_output
    
    @staticmethod
    def from_base_model(model:BaseModel,
                        step_update_begin:int=0,
                        step_use_tsdf:int=0,
                        step_per_update:int = 0,
                        ) -> 'ModelWithTSDF':
        tsdfsampler = TSDFSampler(base_sampler=model.sampler)
        tsdf_model = ModelWithTSDF(tsdfsampler,model.network,
                                   step_update_begin,
                                   step_use_tsdf,
                                   step_per_update)
        return tsdf_model
        
    def export_mesh(self,scale:float=None,offset:np.ndarray=None):
        return self.sampler.export_tsdf_mesh(scale,offset)
    
    def forward_sample_distribution(self,raydata:RayBatch):
        if self.sampler.use_tsdf:
            if self.sampler.use_adaptive: dist_output = self._forward_sample_distribution_adaptive(raydata) 
            else:
                num_samples, num_fine_samples = self.sampler.base_sampler.num_samples,self.sampler.base_sampler.num_fine_samples
                near,far = self.sampler.carving_empty_space(raydata,self.training)
                raydata.near = near * (1. - self.margin_ratio)
                # raydata.far = far * (1. + self.margin_ratio)
                raydata.far = far
                self.sampler.base_sampler.num_samples = num_samples // self.num_samples_divide_factor
                self.sampler.base_sampler.num_fine_samples = num_fine_samples // self.num_fine_samples_divide_factor
                dist_output = super().forward_sample_distribution(raydata)
                self.sampler.base_sampler.num_samples = num_samples 
                self.sampler.base_sampler.num_fine_samples = num_fine_samples
            dist_output["tsdf"] = self.sampler.forward_tsdf(dist_output["position"])
            return dist_output
        else: return super().forward_sample_distribution(raydata)
         
    def forward_tsdf_distribution(self,raydata:RayBatch):
        
        zs = torch.tensor(linear_sample(0.,1.5,128),device=raydata.origin.device).unsqueeze(0).repeat((raydata.batch_size,1)) 
        frustum = Frustum.from_zs(zs)
        samples = raydata.cast_ray(frustum)
        tsdf = self.sampler.forward_tsdf(samples["position"])
        
        return {"tsdf": tsdf,
                "zs": zs}
            
    def _forward_sample_distribution_adaptive(self,raydata:RayBatch):
        if raydata.batch_size != 1: 
            LOG_WARN("sample's raydata should be 1 ray.")
        input_dict,_ = self.sampler.sample(raydata, self.network, self.training)
        raw_output = self.network(input_dict)
        ## volume rendering
        render_output = self.sampler.volume_render_adaptive(input_dict["rays"],\
                            input_dict["deltas"],input_dict["zs"],raw_output=raw_output, out_weight=True)
        if "depth" in render_output:
            render_output["z-depth"] = render_output["depth"] * raydata.depth_scale * raydata.scale
        output_dict = {**raw_output, **render_output}
        output_dict["weight"] = output_dict["weight"].reshape(1,-1) 
        output_dict["frustum"] = Frustum.from_zs(input_dict["zs"].reshape(1,-1))
        output_dict["position"] = input_dict["position"]
        return output_dict