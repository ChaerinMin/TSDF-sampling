import torch
from absl import app, flags
from modules.datamanager import DataManager, NeRFDataset
from models.model import *
from modules.saver import *
from utils.configs import *
from utils.cv_utils.camera import *
# from utils.cv_utils.pose import *
from modules.render import render, render_video
import cv2 as cv

flags.DEFINE_string('gin_configs', None, 'gin config files.')
flags.DEFINE_string('gin_render_configs', None, 'gin config files.')



def main(unused_argv):
    """
    1. load model and datamanager.
    2. choose render camera.
    3. load camera poses. interpolate the poses and shift translation, if necessary
    4. render image, depth given poses
    5. save the result as image files
    6. make video
    """
    train_config = load_train_config(save_config=False)
    config:RenderConfig = load_render_config()
    
    datamanager = DataManager(train_config,1)
    model = load_model()
    
    # load checkpoint
    checkpoint = load_checkpoint(osp.join(config.checkpoint_path,f"{config.checkpoint_step:06d}.pt"))
    model.network.load_state_dict(checkpoint["network"])
    model.sampler.load_state_dict(checkpoint["sampler"])
    model = model.to(0) # zero
    
    tsdf_model = ModelWithTSDF.from_base_model(model)
    tsdf_model.sampler.base_sampler.num_fine_samples = 16
    tsdf_model.sampler.base_sampler.num_samples = 32
    tsdf_model = tsdf_model.to(0) # zero
    tsdf_model.eval()
    depth_file_dir = osp.join(config.checkpoint_path,f"base_sampler_train_view_{50000:06d}") 
    depth_file_fmt = "depth_%05d.npy"
    
    
    # # integrate TSDF    
    for frame_id in datamanager.dataset.frame_ids:
      raydatas, depths = datamanager.get_tsdf_data_with_depth(frame_id=frame_id,batch_size=4096,depth_file_dir=depth_file_dir,depth_file_fmt=depth_file_fmt,device=0)
      for raydata, depth in zip(raydatas,depths):
            tsdf_model.update_tsdf(raydata, depth)
      LOG_INFO(f"frame_id={frame_id}: Completed") 
    
    
    if config.use_novel_cam:
        height, width = config.image_size
        phi = height / width * 180
        novel_cam = EquirectangularCamera({
            "image_size": config.image_size,
            "min_phi_deg": -phi,
            "max_phi_deg": phi,
        })
    else:novel_cam = None
    # model.sampler.num_samples=32
    # model.sampler.num_fine_samples=16
    # render_video(config,model,datamanager,0, novel_cam=novel_cam,video_path=osp.join(config.checkpoint_path,config.video_file))
    tsdf_model.sampler.use_adaptive = True
    tsdf_model.sampler.use_tsdf = True
    tsdf_model.use_resample = False

    render_video(config,tsdf_model,datamanager,0, novel_cam=novel_cam,video_path=osp.join(config.checkpoint_path,config.video_file))

if __name__ == "__main__":
    app.run(main)
