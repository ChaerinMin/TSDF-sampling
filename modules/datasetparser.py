from glob import glob  
import os.path as osp
import numpy as np
from utils.cv_utils.file_utils import *
from utils.cv_utils.tf import Transform
from utils.cv_utils.camera import PinholeCamera
from utils.cv_utils.hybrid_math import normalize 
import cv2 as cv
from utils.logger import *
from enum import Enum

import gin
from dataclasses import dataclass

class DatasetType(Enum):
    MonoSDF = ("MonoSDF", "MonoSDF Dataset type")
    OTHER = ("OTHER", "Other")
    
    @staticmethod
    def from_string(type_str: str)-> 'DatasetType':
        if type_str == 'MonoSDF':
            return DatasetType.MonoSDF
        else:
            return DatasetType.OTHER

@gin.configurable()
@dataclass
class DataParserConfig:

    dataset_dir:str = ""
    # monosdf dataset
    scan_id:int = 1
    center_crop_type:str = ""
    use_mask:bool = False


def parse_dataset(dataset_type_str:str):
    dataset_type = DatasetType.from_string(dataset_type_str)
    config = DataParserConfig()
    if dataset_type == DatasetType.MonoSDF:
        return parse_monosdf_dataset(config)
    else:
        LOG_CRITICAL(f"Invalid DataType: {dataset_type_str}")
        return None

def parse_monosdf_dataset(config:DataParserConfig):
    """
    Parse  monosdf dataset to dictionary
    Adapted https://github.com/autonomousvision/monosdf/blob/main/code/datasets/scene_dataset.py
    """
    dataset_dir = config.dataset_dir
    scan_id = config.scan_id
    center_crop_type = config.center_crop_type
    parser_output = {}
    
    data_dir = osp.join(dataset_dir,f"scan{scan_id}")
    parser_output["base_dir"] =data_dir

    def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
    image_paths = glob_data(osp.join('{0}'.format(data_dir), "*_rgb.png"))
    depth_paths = glob_data(osp.join('{0}'.format(data_dir), "*_depth.npy"))
    normal_paths = glob_data(osp.join('{0}'.format(data_dir), "*_normal.npy"))
    if config.use_mask:
        mask_paths = glob_data(osp.join('{0}'.format(data_dir), "*_mask.npy"))
    else:
        mask_paths = None
    
    parser_output["image_paths"] = image_paths
    parser_output["depth_paths"] = depth_paths
    parser_output["normal_paths"] = normal_paths
    parser_output["mask_paths"] = mask_paths
    parser_output["has_depth"] = True
    parser_output["has_normal"] = True

    
    n_frames = len(image_paths)
    camera_dict = np.load(osp.join(data_dir,"cameras.npz"))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_frames)]
    
    image_size = read_image(image_paths[0]).shape[:2]
    
    cams = []
    cam2worlds = []
    timestamps = []
    indices = {}
    frame_ids = [] 
    
    images = []
    depths = []
    normals = []
    masks = []
    
    for frame_id in range(n_frames):
        indices[frame_id] = frame_id # for monosdf, frame_id and index are same 
        frame_ids.append(frame_id) 
        world_mat = world_mats[frame_id]
        scale_mat = scale_mats[frame_id]
        projection_mat = world_mat @ scale_mat
        
        out = cv.decomposeProjectionMatrix(projection_mat[:3,:4])
        K = out[0]
        R = out[1]
        t = out[2]
        K = K/K[2,2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3,3] = (t[:3] / t[3])[:,0]

        if center_crop_type == 'center_crop_for_replica':
            scale = 384 / 680
            offset = (1200 - 680 ) * 0.5
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'center_crop_for_tnt':
            scale = 384 / 540
            offset = (960 - 540) * 0.5
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'center_crop_for_dtu':
            scale = 384 / 1200
            offset = (1600 - 1200) * 0.5
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'padded_for_dtu':
            scale = 384 / 1200
            offset = 0
            K[0, 2] -= offset
            K[:2, :] *= scale
        elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
            pass
        else:
            raise NotImplementedError

        K = K.tolist()
        cams.append(PinholeCamera.from_K(K,image_size))
        cam2worlds.append(Transform.from_mat(pose))
        timestamps.append(0)
        
        images.append(read_image(image_paths[frame_id],True).reshape(-1,3))
        depths.append(read_npy(depth_paths[frame_id]).reshape(-1,1))
        normal = read_npy(normal_paths[frame_id]).reshape(3,-1)
        normal = normal * 2 - 1 # normalize [0,1] -> [-1,1]
        rot = cam2worlds[-1].rot_mat()
        normal = rot@normal
        normal = normalize(normal,dim=0)
        normals.append(normal.transpose().reshape(-1,3))
        if config.use_mask:
            masks.append(read_npy(mask_paths[frame_id]).bool().reshape(-1,1))
                        
    parser_output["cams"] = cams
    parser_output["cam2worlds"] = cam2worlds
    parser_output["frame_ids"] = frame_ids
    parser_output["indices"] = indices
    parser_output["timestamps"] = timestamps
    
    
    parser_output["images"] = images
    parser_output["depths"] = depths
    parser_output["normals"] = normals
    parser_output["masks"] = masks
    
    return parser_output

# if __name__ == "__main__":
    