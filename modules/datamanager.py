import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from utils.configs import TrainConfig
from utils.logger import * 
from utils.cv_utils.tf import Transform
from utils.cv_utils.camera import Camera
from utils.cv_utils.hybrid_operations import convert_tensor, convert_dict_tensor
from utils.cv_utils.file_utils import read_image, read_npy, read_json
from utils.cv_utils.image_utils import *
from nerf_components.raybatch import RayBatch, DataBatch
from modules.datasetparser import *

import random
import os.path as osp
import numpy as np
from typing import Tuple, Dict, List, Any, Callable
from jaxtyping import Integer
import gin
from dataclasses import dataclass

def split_data(data_dict: Dict[str, Any], split_size: int, data_size: int) -> List[Dict[str, Any]]:
    """
    Splits a data dictionary into smaller sub-dictionaries based on the given split_size. 

    Args:
    - data_dict (Dict[str, Any]): Dictionary containing data arrays or tensors to be split.
    - split_size (int): Desired size for each split.
    - data_size (int): Size of the data arrays or tensors in the first dimension.

    Returns:
    - List[Dict[str, Any]]: List of split data dictionaries.
    """
    # Warning if split_size is larger than the data
    if split_size > data_size:
        LOG_WARN(f"Data size ({data_size}) is smaller than desired split size ({split_size}).")
        return [data_dict]

    split_data_list = []
    num_full_splits = data_size // split_size
    remainder = data_size % split_size

    # Create the full splits
    for i in range(num_full_splits):
        start_index = i * split_size
        end_index = start_index + split_size
        
        split_dict = {key: val[start_index:end_index, :] for key, val in data_dict.items()}
        split_data_list.append(split_dict)

    # Handle any remaining data after the full splits
    if remainder:
        start_index = num_full_splits * split_size
        split_dict = {key: val[start_index:, :] for key, val in data_dict.items()}
        split_data_list.append(split_dict)

    return split_data_list

@gin.configurable()
@dataclass
class DatasetConfig:
    near:float = 0.
    far:float = 1.
    margin:float = 0. # nerf's space bounding box margin (not scale)
    dataset_type_str:str = "MonoSDF" 

class NeRFDataset(Dataset):
    def __init__(self, data_dict:Dict[str,Any],config:DatasetConfig):
        
        self.frame_ids = data_dict["frame_ids"]
        self.indices = data_dict["indices"]
        self.cams:List[Camera] = data_dict["cams"]
        self.cam2worlds = data_dict["cam2worlds"]
        self.timestamps = data_dict["timestamps"]
        
        self.images = data_dict["images"]
        
        self.depths = data_dict["depths"]
        self.normals = data_dict["normals"]
        self.masks = data_dict["masks"]

        self.near:float = config.near
        self.far:float = config.far

        LOG_INFO(f"Total number of images: {len(self.images)}")
        
        self.scale:float = 1.
        self.offset:np.ndarray = np.array([0.,0.,0.])
        
        self.num_pixels:int = self.images[0].shape[0]
        self.train_batch_size:int = self.num_pixels
        self.margin = config.margin
        
        self.has_depth = data_dict["has_depth"]
        self.has_normal = data_dict["has_normal"]
        
    def __len__ (self):
        return len(self.images)
    
    def __getitem__(self, index: Any) -> Tuple[Dict[str,Tensor]]:
        # index = 0 # for debug
        rays,depth_scale = self.cams[index].get_rays(out_scale=True) # (3,n)
        origin, direction = self.cam2worlds[index].get_origin_direction(rays) # (n,3),(n,3)
        # depth_scale = depth_scale.astype(origin.dtype)
        # direction = direction.astype(origin.dtype)
        image = self.images[index] # .astype(origin.dtype)
        depth = self.depths[index] if self.has_depth else None 
        normal = self.normals[index] if self.has_normal else None 
        timestamp = self.timestamps[index]

        sample_idx = self._sample_idx(index)
        
        sample = {
            "origin"     : origin[sample_idx,:],
            "direction"  : direction[sample_idx,:], 
            "timestamp"  : timestamp,
            "frame_index": index,
            "depth_scale": depth_scale[:,sample_idx].transpose().reshape(-1,1),
            "image"      : image[sample_idx,:],
            "depth"      : depth[sample_idx,:] if depth is not None else None,
            "normal"     : normal[sample_idx,:] if normal is not None else None
        }
        return sample
    
    def collate_fn(self, sample):
        sample = convert_dict_tensor(sample[0])

        ones = torch.ones((sample["origin"].shape[0],1))
        near, far =  ones*self.near, ones* self.far

        ray_batch = RayBatch(
                        origin=sample["origin"],
                        direction=sample["direction"],
                        scale=self.scale,
                        offset=convert_tensor(self.offset),
                        timestamps=sample["timestamp"],
                        depth_scale=sample["depth_scale"],
                        frame_index=sample["frame_index"],
                        near=near,
                        far=far)
        batch = DataBatch(color=sample["image"],
                          depth=sample["depth"],
                          normal=sample["normal"])
        return ray_batch, batch

    def _sample_idx(self, index:int) -> Integer[np.ndarray, "batch"]:
        """
        Sample random pixels uniformly across valid pixels of an image.
        Args:
            index: scalar, int, image index.
        Return:
            valid_indices: (batch,), long, uniformly selected indices of valid pixels    
        """
        if len(self.masks) > 0: ## mask
            valid_pixels = self.masks[index].nonzero()[0]
            random_indices = random.sample(range(len(valid_pixels)), k=self.train_batch_size)
            valid_indices = valid_pixels[random_indices]
        else:
            random_indices =  random.sample(list(range(0,self.num_pixels)),self.train_batch_size)
            valid_indices = np.array(random_indices)
        return valid_indices
    
    def get_image_by_frame_id(self, frame_id:int) -> np.ndarray:
        index = self.indices[frame_id]
        cam = self.get_cam_by_frame_id(frame_id)
        return convert_image(self.images[index].reshape(cam.height,cam.width,-1)) 
    
    def get_cam_by_frame_id(self,frame_id:int) -> Camera:
        return self.cams[self.get_index(frame_id)]
    
    def get_cam2world_by_frame_id(self, frame_id:int) -> Transform:
        return self.cam2worlds[self.get_index(frame_id)]

    def get_timestamp_by_frame_id(self,frame_id:int):
        return self.timestamps[self.get_index(frame_id)]  
    
    def get_index(self, frame_id:int) -> int:
        return self.indices[frame_id]

    def get_frame_id(self, index:int) -> int:
        return self.frame_ids[index]

class DataManager:
    def __init__(self, config:TrainConfig, world_size:int):
        self.config = config
        dataset_config = DatasetConfig()
        parser_output = parse_dataset(dataset_config.dataset_type_str)
        self.dataset = NeRFDataset(parser_output,dataset_config)
        self.world_size = world_size
        
        if config.dataset_normalize:
            # compute scale and offset from Pose to normalize
            translations = np.empty((len(self.dataset),3)) # [N,3]
            for idx,pose in enumerate(self.dataset.cam2worlds):
                translations[idx:idx+1,:] = pose.t
            max_x = translations[:,0].max()
            min_x = translations[:,0].min()
            max_y = translations[:,1].max()
            min_y = translations[:,1].min()
            max_z = translations[:,2].max()
            min_z = translations[:,2].min()

            self.dataset.scale = max(max(max_x-min_x,max_y-min_y),max_z-min_z) + self.dataset.margin
            self.dataset.offset = np.array([(max_x+min_x)*0.5,(max_y+min_y)*0.5,(max_z+min_z)*0.5])
        
        LOG_INFO(f"Dataset's scale = {self.dataset.scale} and its offset = \
            ({self.dataset.offset[0]:.3f},{self.dataset.offset[1]:.3f},{self.dataset.offset[2]:.3f})")
            
            
        self.dataset.train_batch_size = config.train_batch_size
        
        self.train_dataloader = DataLoader(self.dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     collate_fn=self.dataset.collate_fn,
                                     num_workers=world_size*4)
        self.iter_train_dataloader = iter(self.train_dataloader)
        
            # self.eval_dataloader = DataLoader(self.dataset,
        #                              batch_size=1,
        #                              shuffle=False,
        #                              collate_fn=self.dataset.collate_fn,
        #                              num_workers=world_size*4)
        # self.iter_eval_dataloader = iter(self.eval_dataloader)
        
    def next_train_ray(self, step:int): 
        try:
            ray_batch, data_batch = next(self.iter_train_dataloader)
        except StopIteration:
            self.iter_train_dataloader = iter(self.train_dataloader)
            ray_batch, data_batch = next(self.iter_train_dataloader)
        
        return ray_batch, data_batch
    
    def get_rendering_train_data(self,frame_id:int, batch_size:int, device:Any) -> List[RayBatch]:
        cam:Camera = self.dataset.get_cam_by_frame_id(frame_id)
        c2w:Transform = self.dataset.get_cam2world_by_frame_id(frame_id)
        timestamp = self.dataset.timestamps[self.dataset.indices[frame_id]]
        return self.get_rendering_data(frame_id,cam,c2w,timestamp,batch_size,device)

    def get_rendering_data(self, frame_id:int, cam:Camera, c2w:Transform,timestamp:int,\
                            batch_size:int,device:Any) -> List[RayBatch]:
        rays, depth_scale = cam.get_rays(out_scale=True)
        origin,direction = c2w.get_origin_direction(rays)
        frame_index = self.dataset.indices[frame_id]
        ones = np.ones_like(depth_scale)
        data = {
            "origin": origin,
            "direction": direction,
            "depth_scale": depth_scale.transpose().reshape(-1,1),
            "near": self.dataset.near * ones.transpose().reshape(-1,1),
            "far": self.dataset.far * ones.transpose().reshape(-1,1),
        }

        data = {key:torch.tensor(val, device=device,dtype=torch.float32) for key,val in data.items()}
        split_data_list = split_data(data,batch_size,origin.shape[0])
        raybatch_list = [ RayBatch(
            origin=sub_data["origin"],
            direction=sub_data["direction"],
            scale=self.dataset.scale,
            offset=convert_tensor(self.dataset.offset,sub_data["origin"]),
            timestamps=timestamp,
            depth_scale=sub_data["depth_scale"],
            frame_index= frame_index,
            near=sub_data["near"],
            far=sub_data["far"]
        ) for sub_data in split_data_list]

        return raybatch_list, (cam.height,cam.width)

    def get_tsdf_data_with_depth(self,frame_id:int, batch_size:int, depth_file_dir:str, depth_file_fmt:str, device:Any):
        """
        
        """
        batchlist, _ = self.get_rendering_train_data(frame_id,batch_size,device)
        index = self.dataset.get_index(frame_id)
        depth = read_npy(osp.join(depth_file_dir,depth_file_fmt%index)).reshape(-1,1)
        depth_split = split_data({"depth": depth},batch_size,depth.shape[0])
        depth_split = [torch.tensor(subdata["depth"],dtype=torch.float32,device=device) for subdata in depth_split]
        return batchlist, depth_split
    
    def get_plot_data(self,pt2d:Tuple[int],frame_id:int, cam:Camera, c2w:Transform,timestamp:int,device:Any):
        uv = np.array([[pt2d[0]],[pt2d[1]]])        
        ray, depth_scale = cam.get_rays(uv=uv,out_scale=True)
        origin,direction = c2w.get_origin_direction(ray)
        frame_index = self.dataset.indices[frame_id]
        raydata = RayBatch(
            origin=torch.tensor(origin,device=device,dtype=torch.float32),
            direction=torch.tensor(direction,device=device,dtype=torch.float32),
            scale=self.dataset.scale,
            offset=torch.tensor(self.dataset.offset,device=device,dtype=torch.float32),
            timestamps=timestamp,
            depth_scale=torch.tensor(depth_scale.transpose().reshape(-1,1),device=device,dtype=torch.float32),
            frame_index= frame_index,
            near= self.dataset.near * torch.ones_like(depth_scale),
            far=self.dataset.far * torch.ones_like(depth_scale)
        )
        return raydata
    
    def get_image_by_frame_id(self, frame_id:int) -> np.ndarray:
        return self.dataset.get_image_by_frame_id(frame_id)
    
    def get_cam_by_frame_id(self,frame_id:int) -> Camera:
        return self.dataset.get_cam_by_frame_id(frame_id)
    
    def get_cam2world_by_frame_id(self, frame_id:int) -> Transform:
        return self.dataset.get_cam2world_by_frame_id(frame_id)

    def get_timestamp_by_frame_id(self,frame_id:int):
        return self.dataset.get_timestamp_by_frame_id(frame_id)
    
    def get_index(self, frame_id:int) -> int:
        return self.dataset.get_index(frame_id)

    def get_frame_id(self, index:int) -> int:
        return self.dataset.get_frame_id(index)

    @property
    def size(self) -> int:
        return len(self.dataset)
    
if __name__ == "__main__":
    LOG_INFO("Check DataManager")
    config = TrainConfig()

