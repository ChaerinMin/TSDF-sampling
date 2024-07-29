import torch
from nerf_components.raybatch import RayBatch
from nerf_components.raybatch import RayBatch
from models.model import BaseModel
from modules.datamanager import NeRFDataset, DataManager
from utils.cv_utils.hybrid_operations import convert_numpy, concat
from utils.cv_utils.file_utils import *
from utils.cv_utils.image_utils import *
from utils.cv_utils.camera import *
from utils.cv_utils.tf import *
from utils.logger import *
from utils.configs import RenderConfig
import numpy as np
from typing import Dict, List, Any,Tuple
import os.path as osp
import os
from time import time
import cv2 as cv


def render_video(config:RenderConfig,model:BaseModel,
                 datamanager:DataManager,
                 device:Any,novel_cam:Camera=None,video_path:str=""):
    
    output_dir = osp.join(config.checkpoint_path,f"{config.video_file[:-4]}_{config.checkpoint_step:06d}") 
    os.makedirs(output_dir,exist_ok=True)
    num_train_frames = len(datamanager.dataset.frame_ids) 
    include_normal:bool = False

    video_frame_cnt:int = 0
    total_video_frames:int = (num_train_frames-1) * config.interpolation_times + 1

    render_infos = []
    for frame_id in datamanager.dataset.frame_ids:
        render_infos += interpolate_poses_and_info(frame_id,datamanager.dataset,config.interpolation_times)
    
    for render_info in render_infos:
        frame_id = render_info["frame_id"]
        cam = datamanager.get_cam_by_frame_id(frame_id) if novel_cam is None else novel_cam
        c2w = render_info["c2w"]
        timestamp = render_info["timestamp"]
        raydatas, image_size = datamanager.get_rendering_data(frame_id,cam,c2w,timestamp,config.render_batch_size,device)
        render_output = render(raydatas,model,image_size) # render_output = {image, depth}
        image = convert_image(render_output["rgb"]) 
        depth = float_to_image(render_output["depth"],config.depth_min_val,config.depth_max_val) 
        write_image(image, osp.join(output_dir,f"image_{video_frame_cnt:06d}.png"))
        write_npy(render_output["depth"], osp.join(output_dir,f"depth_{video_frame_cnt:06d}.npy"))
        write_npy(render_output["acc"], osp.join(output_dir,f"acc_{video_frame_cnt:06d}.npy"))
        write_image(depth, osp.join(output_dir,f"depth_{video_frame_cnt:06d}.png"))
        if "normal" in render_output:
            include_normal = True
            normal = normal_to_image(render_output["normal"])
            write_image(normal, osp.join(output_dir,f"normal_{video_frame_cnt:06d}.png"))
        video_frame_cnt += 1
        LOG_INFO(f"{video_frame_cnt}/{total_video_frames}: completed")
            
    cam = datamanager.get_cam_by_frame_id(0) if novel_cam is None else novel_cam
    height,width = cam.height,cam.width

    if novel_cam is not None:
        video_shape = (width,3*height) if include_normal else  (width,2*height)
    else:
        video_shape = (3*width,height) if include_normal else  (2*width,height)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(video_path,
                         fourcc=fourcc,fps=config.fps,frameSize=video_shape)
    # make video
    LOG_INFO("Loading images....")
    for i in range(total_video_frames):
        concat_image = read_and_concat_image(config,output_dir,i,include_normal,vertical_concat=(novel_cam is not None))
        concat_image = cv.cvtColor(concat_image,cv.COLOR_RGB2BGR)
        out.write(concat_image)
    LOG_INFO("Making video....")
    out.release()
    LOG_INFO("Completed to make video")

def read_and_concat_image(config:RenderConfig,output_dir:str,index:int, include_normal:bool, vertical_concat):
    image = read_image(osp.join(output_dir, f"image_{index:06d}.png"))
    depth = float_to_image(1./read_npy(osp.join(output_dir, f"depth_{index:06d}.npy")),1./config.depth_max_val,1./config.depth_min_val if config.depth_min_val > 0. else 3.)
    dim = 0 if vertical_concat else 1
    concat_image = concat([image,depth],dim=dim)
    if include_normal:
        normal = read_image(osp.join(output_dir, f"normal_{index:06d}.png"))
        concat_image = concat([concat_image,normal],dim=dim)
    return concat_image

def interpolate_poses_and_info(frame_id:int,dataset:NeRFDataset, interpolation_times:int=1):
    if frame_id == dataset.frame_ids[-1]:
        # last frame: no interpolation
        idx = dataset.get_index(frame_id) 
        return [
            {   "frame_id": frame_id,
                "c2w": dataset.get_cam2world_by_frame_id(frame_id),
                "timestamp": dataset.get_timestamp_by_frame_id(frame_id),
                "frame_index": dataset.get_index(frame_id),
                "t": 0 
            }]
    
    interpolate_info = []
    for time in range(interpolation_times):
        t = time / interpolation_times
        idx = dataset.get_index(frame_id) 
        p1:Transform = dataset.cam2worlds[idx]
        p2:Transform = dataset.cam2worlds[idx+1]
        c2w:Transform = interpolate_transform(p1,p2,t)
        tp = int((dataset.timestamps[idx]+dataset.timestamps[idx+1]) *0.5)
        interpolate_info.append({
            "frame_id": frame_id,
            "c2w": c2w,
            "timestamp": tp,
            "frame_index": idx,
            "t": t
        })
    return interpolate_info


def render(
        raydatas:List[RayBatch],
        model:BaseModel,
        image_size:Tuple[int,int],
        out_meta:bool=False
        ) -> Dict[str, np.ndarray]:
    
    rgb = []
    depth = []
    acc = []
    
    model.eval()
    tic = time()
    for _,raydata in enumerate(raydatas):
        render_output = model(raydata)
        rgb.append(convert_numpy(render_output["rgb"]))
        depth.append(convert_numpy(render_output["depth"]))
        acc.append(convert_numpy(render_output["acc"]))
        
    toc = time()
    LOG_INFO(f"Rendering time: {toc-tic:04f}")
    render_dict = {
        "rgb": np.concatenate(rgb, axis=0).reshape(image_size[0],image_size[1],3),
        "depth": np.concatenate(depth, axis=0).reshape(image_size[0],image_size[1],1),
        "acc": np.concatenate(acc, axis=0).reshape(image_size[0],image_size[1],1)
    }

    if out_meta:
        render_dict["render_time"] = toc - tic    
    return render_dict