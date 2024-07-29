import torch
import numpy as np
import torch.nn as nn
from nerf_components.encoding import *
from nerf_components.encoding import Dict, Tensor
from nerf_components.mlp import * 
from nerf_components.density import *
from nerf_components.mlp import Tensor 
from nerf_components.network import *
from nerf_components.network import Dict, Tensor
from nerf_components.raybatch import *
from dataclasses import dataclass
from typing import Any
from jaxtyping import Float
import extensions.raymarching as raymarching
from nerf_components.raybatch import Dict, Float, Frustum, Tensor       
import mcubes
import open3d as o3d


@gin.configurable
def load_sampler(sampler_cls):
     return sampler_cls()

def linear_sample(near:float, far:float, num_samples:int) -> np.ndarray:
    unit_zs = np.linspace(start=0.,stop=1.,num=num_samples, dtype=np.float32)
    return near* (1. - unit_zs) + far* unit_zs

# (https://github.com/bmild/nerf/)
def hierarchical_sample(
        bins: Tensor, weights: Tensor, num_samples: int, pertube: bool = False) \
            -> Tensor:
    batch_size = weights.shape[0]
    device = bins.device
    pdf = weights / (torch.sum(weights, -1, keepdim=True) + 1e-6)
    cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    if pertube:
        u = torch.rand([batch_size, num_samples], device=device)
    else:
        u = torch.linspace(0.05, 0.95, steps=num_samples, device=device)
        u = u.expand([batch_size, num_samples]).contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, num_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1] - cdf_g[...,0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[...,0]) / denom
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    return samples


class Sampler(nn.Module):
    def sample(self, raybatch:RayBatch) -> Tensor:
        raise NotImplementedError
    
    def _sample_coarse_frustum(self,near:Float[Tensor,"batch 1"],far:Float[Tensor,"batch 1"] ,
                       pertub:bool=True) -> Frustum:
        device = near.device
        unit_sample = linear_sample(0.,1., self.num_samples+1)
        unit_sample = torch.tensor(unit_sample, device=device) # (num_samples+1,)
        unit_sample = unit_sample.unsqueeze(0).repeat(near.shape[0],1) # (batch,num_samples+1)
        if pertub:
            mid_unit_sample = 0.5 * (unit_sample[:, 1:] + unit_sample[:, :-1])
            upper_unit_sample = torch.cat([mid_unit_sample, unit_sample[:, -1:]], -1)
            lower_unit_sample = torch.cat([unit_sample[:, :1], mid_unit_sample], -1)
            perturb_rand = torch.rand(unit_sample.shape, device=device, requires_grad=False)
            zs = lower_unit_sample + perturb_rand * (upper_unit_sample - lower_unit_sample)
        zs = near* (1. - unit_sample) + far*unit_sample
        frustum = Frustum(start=zs[:,:-1], end= zs[:,1:])
        return frustum

    def _compute_weight(self, density:Float[Tensor,"batch num_samples"],\
                          frustum:Frustum) -> Float[Tensor,"batch num_samples"]:
        if len(density.shape) == 2: # [batch*num_samples,1]
            batch_size = frustum.batch_size
            density = density.reshape(batch_size,-1)
        delta = frustum.deltas
        alpha = 1. - torch.exp(-density*delta)
        trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1.-alpha],-1),dim=-1)[:,:-1]
        weight = alpha * trans
        return weight
    
    def _composite(self, weight:Float[Tensor,"batch num_samples"],\
                    ray_output:Float[Tensor,"batch num_samples size"],
                    normalize:bool=False) -> Float[Tensor,"batch size"]:
        if len(ray_output.shape) == 2: # [batch*num_samples,size]
            batch_size, num_samples = weight.shape[0:2]
            ray_output = ray_output.reshape(batch_size,num_samples,-1)
        composited_output = torch.sum(weight.unsqueeze(-1) * ray_output, dim=1)# [batch size]
        if normalize:
            composited_output = composited_output / (weight.sum(dim=1,keepdim=True) + 1e-8)
        return composited_output
    
    def volume_render(self, frustum:Frustum,raw_output:Dict[str,Tensor], out_weight:bool=False) -> Dict[str,Tensor]:
        density = raw_output["density"]
        weight = self._compute_weight(density=density,frustum=frustum)
        render_output = {}
        render_output["rgb"] = self._composite(weight,raw_output["raw_rgbs"],False) # rgb
        render_output["depth"] = self._composite(weight, frustum.zs, True) # depth
        if "raw_normals" in raw_output: # normal
            render_output["normal"] = self._composite(weight,raw_output["raw_normals"])
        # weight accum
        render_output["acc"] = weight.sum(dim=1, keepdim=True)
        if out_weight: 
            render_output["weight"] = weight
            # compute num important samples
            # sorted_weight,_ = torch.sort(weight/weight.sum(dim=1, keepdim=True),dim=1,descending=True)
            # cum_sum_weight = torch.cumsum(sorted_weight,dim=1)
            # important_samples_cnt = torch.sum(cum_sum_weight < 0.99, dim=1)
            # render_output["num_important_samples"] = important_samples_cnt.sum().item()
        return render_output

@gin.configurable
class PdfSampler(Sampler):
    def __init__(self,
                num_samples:int =  64, # coarse samples
                num_fine_samples:int = 128, # fine samples
                sample_network_fn:Callable = None ,  
                use_mip:bool = False):
        super(PdfSampler,self).__init__()
        self.num_samples = num_samples
        self.num_fine_samples = num_fine_samples
        self.sample_network = sample_network_fn() if sample_network_fn is not None else None
        self.use_mip = use_mip
    
    def sample(self, raybatch:RayBatch,network:BaseNetwork,training:bool=True, near:Float[Tensor,"batch 1"]=None, far:Float[Tensor,"batch 1"]=None):
        """
            sampling ray
            1. sample coarsely
            2. evaluate
            3. sample fine and merge
        """
        if near is None or far is None: near,far = raybatch.near_far(use_aabb=False)
        
        frustum = self._sample_coarse_frustum(near,far,training)
        if self.use_mip:
            coarse_input = raybatch.cast_ray_mip(frustum)
        else:
            coarse_input = raybatch.cast_ray(frustum)
        
        if self.sample_network and training:
            coarse_output = self.sample_network(coarse_input)
            density = coarse_output["density"]
        else:
            with torch.no_grad():
                if self.sample_network: density = self.sample_network.get_density(coarse_input) 
                else: density = network.get_density(coarse_input)
        weight = self._compute_weight(density,frustum)
        
        fine_frustum = self._sample_fine_frustum(frustum, weight,perturb=training)
        if self.use_mip:
            sample_output = raybatch.cast_ray_mip(fine_frustum)
        else:
            sample_output = raybatch.cast_ray(fine_frustum, self.use_mip)
        # training sample network
        extra_output = {}
        if self.sample_network and training:
            extra_output["coarse_rgb"] = self._composite(weight,coarse_output["raw_rgbs"])
        return sample_output, extra_output
    
    def _sample_fine_frustum(self, frustum:Frustum,\
                    weights:Float[Tensor,"batch num_samples"],
                    perturb:bool=False) \
     -> Frustum:
         # resampling
        bins = frustum.zs
        with torch.no_grad():
            fine_zs = hierarchical_sample(bins,weights,self.num_fine_samples,perturb)
        combined_zs, _ = torch.sort(torch.cat([frustum.zs, fine_zs],-1),-1)
        fine_frustum = Frustum.from_zs(combined_zs)      
        return fine_frustum

'''
    TSDF and Weight Grid
    TSDF Grid is normalized as bound ([-bound,bound]^3 -> [-1,1]^3)
    truncated distance: voxel_size * trunc
    narrow_distance: voxel_ * narrow
    
    weight_type: int, scalar, weight type
        0: constant weight
        1: linear_weight
        2: narrow_linear_weight 
        3: narrow_exp_weight 
    weighting functions: See Fig 5. of Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions (RSS 2013)
'''
@gin.configurable
class TSDFSampler(Sampler):
    def __init__(self,
                 base_sampler:Sampler,
                 use_tsdf:bool = False,
                 use_adaptive:bool = False,
                 num_tsdf_samples:int = 0,
                 grid_size:int = 512,
                 trunc:int = 39, # delta of the weighting function for updating the TSDF. For weighting function, see the docs above.
                 narrow: int = 13, # epsilon of the weighting function for updating the TSDF.
                 surf:int = 27, # where to decide the near bound t_n (in factor of voxel_size). This is the D_s in our Algorithm 1.
                 weight_type: int = 1,
                 nb_margin:int = 3, # how many neighbor voxel layers to consider for the far bound t_f. This makes NeighborVoxels in our Algorithm 1.
                 max_walk_step:int = 15, # how many ray marching steps needed to say to stop before finally deciding the far bound t_f. This is the M in our Algorithm 1.
                 min_dt:float = 1.,
                 max_dt:float = 1.2,
                 # After deciding t_n and t_f, the samplings are decided by the step size dt, if use adaptive. min_dt and max_dt make the dt.
                 ):
        super(TSDFSampler,self).__init__()
        
        self.base_sampler = base_sampler
        self.num_tsdf_samples = num_tsdf_samples
        self.grid_size = grid_size
        self.register_buffer('tsdf_grid', -torch.ones(grid_size**3))
        self.register_buffer('weight_grid', torch.zeros(grid_size**3))
               
        self.voxel_size = 2. / self.grid_size
        self.trunc = self.voxel_size * trunc
        self.narrow = self.voxel_size * narrow
        self.surf = self.voxel_size * surf
        
        self.weight_type = weight_type
        
        self.min_dt = self.voxel_size * min_dt
        self.max_dt = self.voxel_size * max_dt
        self.nb_margin = nb_margin
        self.max_walk_step = max_walk_step
        self.use_tsdf = use_tsdf    
        self.use_adaptive = use_adaptive
    
    def update_tsdf_grid(self, origin:Float[Tensor,"batch 3"], direction:Float[Tensor,"batch 3"], depth:Float[Tensor,"batch 1"]):
        '''
        integrate TSDF and Weight Grid with depths regarding ray's origin and direction
        Args
            origin: float, [B,3], camera origin
            direction: float, [B,3], camera direction
            depth: float, [B,1], z-depth            
        '''
        raymarching.update_tsdf(origin,direction,depth, self.grid_size, self.trunc, self.narrow, self.tsdf_grid, self.weight_grid, self.weight_type)
        return
    
    def _march_rays(self, origin:Float[Tensor,"batch 3"], direction:Float[Tensor,"batch 3"], training:bool):
        rays, xyzs, dirs, deltas, zs, intervals = raymarching.march_rays_tsdf_uniform(origin, direction,1.,self.tsdf_grid, self.surf,
                        (self.min_dt + self.max_dt)*0.5, self.nb_margin, self.max_walk_step, self.grid_size, None, None, 512,training)
        return rays, xyzs, dirs, deltas, zs, intervals
    
    def _forward_from_tsdf(self,raybatch:RayBatch,network:BaseNetwork,training:bool) -> Frustum:
        origin,direction = raybatch.get_origin_direction(normalize=True)
        rays, xyzs, dirs, deltas,zs, intervals = self._march_rays(origin,direction,training)
        if self.num_tsdf_samples > 0:
            density = network.get_density({'position': xyzs})
            alphas = 1 - torch.exp(-density * deltas[:,0:1])
            nears = intervals[:,0] 
            rays, xyzs, dirs, deltas,zs, zs_fine, weights_sum = raymarching.evaluate(origin, direction, rays, alphas, deltas, nears, self.num_tsdf_samples)
        sample_output = {
            "position": xyzs,
            "direction": dirs,
            "rays": rays,
            "deltas": deltas,
            "zs": zs,
        }
        if self.num_tsdf_samples > 0:
            extra_output = {
                "zs_fine": zs_fine,
                "weights_sum": weights_sum
            }
        else:
            extra_output = {}
            
        return sample_output, extra_output
    
    def sample(self,raybatch:RayBatch,network:BaseNetwork,training:bool):
        if self.use_adaptive:
            with torch.no_grad(): 
                return self._forward_from_tsdf(raybatch,network,training)
        else: return self.base_sampler.sample(raybatch,network,training)
    
    def _composite(self, weight: Tensor, ray_output: Tensor, normalize: bool = False) -> Tensor:
        return self.base_sampler._composite(weight, ray_output, normalize)
    
    def _compute_weight(self, density: Tensor, frustum: Frustum) -> Tensor:
        return self.base_sampler._compute_weight(density, frustum)
    
    def volume_render(self, frustum: Frustum, raw_output: Dict[str, Tensor],out_weight:bool=False) -> Dict[str, Tensor]:
        return self.base_sampler.volume_render(frustum, raw_output,out_weight)    
    
    def volume_render_adaptive(self,rays:Tensor,deltas:Tensor,zs:Tensor, raw_output:Dict[str, Tensor],out_weight=False) -> Dict[str, Tensor]:
        weights, weights_sum, depth, image = \
            raymarching.volume_render_adaptive(raw_output["density"],raw_output["raw_rgbs"],zs, deltas, rays)
        render_output = {
            "rgb": image,
            "depth": depth,
            "acc": weights_sum
        }
        if out_weight: render_output["weight"] = weights
        return render_output
        
    def export_tsdf_mesh(self, scale:float=None,offset:np.ndarray=None):
        xs = torch.arange(0, self.grid_size)
        ys = torch.arange(0, self.grid_size)
        zs = torch.arange(0, self.grid_size)
        coords = torch.stack(
            torch.meshgrid([xs, ys, zs], indexing='ij'), dim=-1).to(self.tsdf_grid.device)
        coords = coords.permute(
                (0, 2, 1, 3)).contiguous().reshape((-1, 3)).long() 
        
        inds = raymarching.morton3D(coords)
        
        sdfs = torch.index_select(self.tsdf_grid,0,inds)  
        sdfs = sdfs.reshape(self.grid_size,self.grid_size,self.grid_size).permute(0,2,1)
        sdfs = sdfs.detach().cpu().numpy()
        vertices, triangles = mcubes.marching_cubes(-sdfs,0)
        vertices = (vertices / self.grid_size -1.)
        if offset is not None:
            vertices = vertices + offset
        if scale is not None:
            vertices *= scale
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        return mesh
    
    def forward_tsdf(self, xyzs:Tensor) -> Tensor:
        coords = torch.clamp(0.5*(xyzs + 1.)*(self.grid_size-1),0.,self.grid_size-1).long()
        inds = raymarching.morton3D(coords)
        tsdfs = torch.index_select(self.tsdf_grid,0,inds)
        return tsdfs
    
    def carving_empty_space(self,raybatch:RayBatch, training:bool, out_mean_deltas:bool=False):
        origin, direction = raybatch.get_origin_direction(normalize=True)
        _, _, _, deltas,_, intervals = self._march_rays(origin,direction,training)
        near = intervals[:,0:1] * raybatch.scale
        # near = intervals[:,0:1]
        far = intervals[:,1:2] * raybatch.scale
        # far = intervals[:,1:2]
        mean_deltas = torch.mean(deltas).item()
        
        if out_mean_deltas: return near,far, mean_deltas
        else: return near,far
    
    def reset(self):
        self.tsdf_grid.fill_(-1.)
        self.weight_grid.fill_(0.)