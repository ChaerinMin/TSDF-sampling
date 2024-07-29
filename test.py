import torch
import gc
from absl import app, flags
from modules.datamanager import DataManager
from models.model import *
from modules.saver import *
from utils.configs import *
from utils.cv_utils.camera import *
from utils.cv_utils.image_utils import *
from modules.loss.losses import * 
from utils.cv_utils.file_utils import write_image, write_npy
from modules.render import render, render_video


flags.DEFINE_string('gin_configs', "", 'gin config files.')

def main(unused_argv):

    config = load_train_config(save_config=False)
    
    datamanager = DataManager(config,1)
    model = load_model()
    checkpoint_path = config.checkpoint_path
    # load checkpoint
    checkpoint = load_checkpoint(osp.join(checkpoint_path,f"{50000:06d}.pt"))
    model.network.load_state_dict(checkpoint["network"])
    model.sampler.load_state_dict(checkpoint["sampler"])
    model = model.to(0)
    model.eval()
    tsdf_model = ModelWithTSDF.from_base_model(model)
    tsdf_model = tsdf_model.to(0) # zero
    tsdf_model.eval()
    depth_file_dir = osp.join(checkpoint_path,f"base_sampler_train_view_{50000:06d}") 
    depth_file_fmt = "depth_%05d.npy"
    
    # depth_file_dir = "/home/sehyun/replica/scan1"
    # depth_file_fmt = "%06d_pred_depth.npy"
    
    # integrate TSDF    
    if config.use_tsdf: 
        for frame_id in datamanager.dataset.frame_ids:
            raydatas, depths = datamanager.get_tsdf_data_with_depth(frame_id=frame_id,batch_size=4096,depth_file_dir=depth_file_dir,depth_file_fmt=depth_file_fmt,device=0)
            for raydata, depth in zip(raydatas,depths):
                # ray_depth = depth / (raydata.depth_scale * raydata.scale) # (batch, 1)
                tsdf_model.update_tsdf(raydata, depth)
            LOG_INFO(f"frame_id={frame_id}: Completed")
        render_model = tsdf_model
    else:
        render_model = model

    psnrs = []
    ssims = []
    render_times = []
    
    for frame_id in datamanager.dataset.frame_ids:
        batchlist, image_size = datamanager.get_rendering_train_data(frame_id,config.render_batch_size,0)
        render_output = render(batchlist, render_model, image_size,True)
        image = render_output["rgb"].reshape(-1,3)
        index = datamanager.dataset.indices[frame_id]
        gt_image = datamanager.dataset.images[index]
        
        psnrs.append(compute_psnr(torch.tensor(image),torch.tensor(gt_image)).item()) 
        ssims.append(compute_ssim(torch.tensor(image),torch.tensor(gt_image)).item())
        render_times.append(render_output["render_time"])
        
        if frame_id % 10 == 0:
            LOG_INFO(f"frame_id: {frame_id}, PSNR: {psnrs[-1]}")
    
    height = datamanager.get_cam_by_frame_id(0).height 
    width = datamanager.get_cam_by_frame_id(0).width 
    
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    render_times = np.array(render_times)
    
    
    LOG_INFO(f"mean PSNR:{psnrs.mean():.3f}")
    LOG_INFO(f"mean SSIM:{ssims.mean():.5f}")
    LOG_INFO(f"IMG_SIZE: {width}*{height} mean RENDER TIME: {render_times.mean():.3f}")
        
if __name__ == "__main__":
    app.run(main)
