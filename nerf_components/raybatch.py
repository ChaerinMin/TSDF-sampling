import torch
from torch import Tensor
from typing import Any, Tuple, Dict
from utils.logger import LOG_ERROR
from dataclasses import dataclass
from jaxtyping import Float

@dataclass
class Frustum:
    start:Tensor # [B,N]
    end:Tensor # [B,N]

    @staticmethod
    def from_zs(zs:Float[Tensor, "batch num_samples"]):
        deltas = zs[:,1:] - zs[:,:-1] # [B,N-1]
        deltas = torch.cat([deltas, deltas[:,-2:-1]], dim=-1) # [B,N]
        return Frustum(start=zs-0.5*deltas,end=zs+0.5*deltas)
    @property 
    def zs(self) -> Tensor:
        return (self.start + self.end) * 0.5
    @property 
    def deltas(self) -> Tensor:
        return self.end - self.start
    @property
    def bins(self) -> Tensor:
        return torch.cat([self.start, self.end[:,-2:-1]])
    @property
    def batch_size(self) -> int:
        return self.start.shape[0]
    

@dataclass
class RayBatch:
    origin:Tensor # [B,3], origin of ray
    direction:Tensor # [B,3], directions of ray
    # origin = scale * normalized_origin + offset
    scale:float # scale of origin 
    offset:Tensor # offset of origin
    depth_scale:Tensor # z_depth = depth_scale * ray_depth
    frame_index:int # frame index
    near:Tensor # [B,1], near of ray
    far:Tensor # [B,1], far of ray
    timestamps:int = 0 # timestamps

    def __getitem__(self,mask):
        return RayBatch(
            origin=self.origin[mask],
            direction=self.direction[mask],
            scale=self.scale,
            offset=self.offset,
            depth_scale = self.depth_scale[mask],
            frame_index=self.frame_index,
            near=self.near[mask],
            far=self.far[mask],
            timestamps=self.timestamps
        )
    
    def get_origin_direction(self,normalize:bool=True):
        if normalize: origin = (self.origin - self.offset) / self.scale
        else: origin = self.origin
        return origin, self.direction
    
    def to(self,device:Any):
        self.origin = self.origin.to(device)
        self.direction = self.direction.to(device)
        self.offset = self.offset.to(device)
        self.near = self.near.to(device)
        self.far = self.far.to(device)
        self.depth_scale = self.depth_scale.to(device)
    
    def denormalize(self, x:Tensor) -> Tensor:
        if len(x.shape) == 1: return self.scale * x # [B]
        elif len(x.shape) == 2:
            if x.shape[-1] == 1: return self.scale * x # [B,1]
            elif x.shape[-1] == 3: return self.scale * x + self.offset # [B,3]
        else: 
            LOG_ERROR("Invalid Shape Error")
            return None
    
    def near_far(self,normalize:bool=True, use_aabb:bool=False) -> Tuple[Tensor,Tensor]:
        if use_aabb: near,far = self.aabb()
        else: near, far = self.near, self.far
        if normalize: return near / self.scale, far / self.scale
        else: near, far

    def cast_ray(self, frustum:Frustum, out_latent:bool=False, normalize:bool=True) -> Dict[str,Tensor]:
        """
        Casts a ray given a set of distances.

        Args:
        - zs (Tensor): A tensor containing the distances at which to sample along the ray. 
                       Expected shape is either [batch_size, num_samples] or [batch_size, num_samples, 1].
        - out_latent(bool): Whether to include the latent index in the output dictionary. Default is False.
        - normalize(bool): Whether to include the latent index in the output dictionary. Default is Tue.
        Returns:
        - Dict[str, Tensor]: A dictionary containing the sampled positions and directions. 
        """
        zs = frustum.zs
        batch_size = zs.shape[0]
        num_samples = zs.shape[1]
        ray_dict = {}
        origin, direction = self.get_origin_direction(normalize)
        position = origin.unsqueeze(1) + zs.unsqueeze(2) * direction.unsqueeze(1)
        ray_dict["position"] = position.reshape(-1,3)
        ray_dict["direction"] = direction.unsqueeze(1).repeat(1,num_samples,1).reshape(-1,3)
        ray_dict["frustum"] = frustum
        if out_latent:
            ray_dict["latent_index"] = self.frame_index * torch.ones(batch_size*num_samples,\
                dtype=torch.int64, device=self.origin.device)
         
        return ray_dict
    
    @property
    def batch_size(self) ->int:
        return self.origin.shape[0]
    
    def aabb(self) -> Tuple[Tensor,Tensor]:
        """
        Compute the near and far intersection points of rays with an axis-aligned cube.

        Args:
        - origin (torch.Tensor): Origins of the rays with shape [B, N, 3].
        - direction (torch.Tensor): Directions of the rays with shape [B, N, 3].
        - bound (float): Half length of the cube's side. The cube's extents are from -bound to bound in all axes.

        Returns:
        - near (torch.Tensor): The near intersection distances with shape [B, N, 1].
        - far (torch.Tensor): The far intersection distances with shape [B, N, 1].
        """

        # Calculate potential near (tmin) and far (tmax) intersections with cube's planes for each ray.
        # To prevent division by zero, we add a tiny constant (1e-15) to the ray directions.
        tmin = (-self.scale - self.origin) / (self.direction + 1e-15)
        tmax = (self.scale - self.origin) / (self.direction + 1e-15)

        # The intersections might be swapped depending on the ray's direction.
        # Swap them back if necessary.
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]

        # Check for rays that miss the cube. If the far intersection is less than the near one,
        # the ray doesn't intersect the cube. In this case, set both near and far to a large value.
        no_intersection_mask = far < near
        near[no_intersection_mask] = 1e9
        far[no_intersection_mask] = 1e9

        # Clamp near and far values to ensure they lie within a valid range.
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far
    
@dataclass
class DataBatch:
    color: Tensor # [B,3], rgb color
    depth: Tensor # [B,1], depth
    normal: Tensor # [B,3], surface normal
    
    def to(self,device:Any):
        self.color = self.color.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.normal is not None:
            self.normal = self.normal.to(device)
            
    def __getitem__(self,mask):
        return DataBatch(
            color = self.color[mask],
            depth = self.depth[mask] if self.depth is not None else None,
            normal = self.normal[mask] if self.normal is not None else None,
        )
            