import torch
from absl import app, flags
from modules.datamanager import DataManager
from models.model import *
from modules.saver import *
from utils.configs import *
from utils.cv_utils.camera import *
from utils.cv_utils.image_utils import *
from utils.cv_utils.file_utils import write_image, write_npy
from modules.render import render, render_video # , render_tsdf_acc


flags.DEFINE_string('gin_configs', "configs/train_vanilla_nerf_wo_coarse_unnorm.gin", 'gin config files.')

def main(unused_argv):
    """
    1. load model and datamanager.
    2. choose render camera.
    3. load camera poses. interpolate the poses and shift translation, if necessary
    4. render image, depth given poses
    5. save the result as image files
    6. make video
    """
    config = load_train_config(False)
    config.dataset_normalize = False
    
    datamanager = DataManager(config,1)
    model = load_model()
    checkpoint_path = config.checkpoint_path
    # load checkpoint
    checkpoint = load_checkpoint(osp.join(checkpoint_path,f"{50000:06d}.pt"))
    model.network.load_state_dict(checkpoint["network"])
    model.sampler.load_state_dict(checkpoint["sampler"])
    tsdf_model = ModelWithTSDF.from_base_model(model)
    tsdf_model = tsdf_model.to(0) # zero
    tsdf_model.eval()
    depth_file_dir = osp.join(checkpoint_path,f"base_sampler_train_view_{50000:06d}") 
    depth_file_fmt = "depth_%05d.npy"
    
    # save depths
    if not osp.join(depth_file_dir, depth_file_fmt % 0):
        tsdf_model.sampler.base_sampler.num_samples //= 2  
        tsdf_model.sampler.base_sampler.num_fine_samples //= 2  
        eval_output = osp.join(checkpoint_path,"tsdf_output_res_512")
        os.makedirs(eval_output,exist_ok=True)
        for frame_id in datamanager.dataset.frame_ids:
            raydatas, image_size = datamanager.get_rendering_train_data(frame_id,4096,0)
            render_output = render(raydatas,tsdf_model,image_size) # render_output = {image, depth}
            image = convert_image(render_output["rgb"])
            depth_image = float_to_image(render_output["depth"],0.05,3.5) 
            write_image(image,osp.join(eval_output,f"image_{frame_id:05d}.png"))
            write_npy(render_output["depth"],osp.join(eval_output,f"depth_{frame_id:05d}.npy"))
            write_image(depth_image,osp.join(eval_output,f"depth_{frame_id:05d}.png"))
            write_image((render_output["acc"]*255).astype(np.uint8),osp.join(eval_output,f"acc_{frame_id:05d}.png"))

    # integrate TSDF    
    for frame_id in datamanager.dataset.frame_ids:
        raydatas, depths = datamanager.get_tsdf_data_with_depth(frame_id=frame_id,batch_size=4096,depth_file_dir=depth_file_dir,depth_file_fmt=depth_file_fmt,device=0)
        for raydata, depth in zip(raydatas,depths):
            # ray_depth = depth / (raydata.depth_scale * raydata.scale) # (batch, 1)
            tsdf_model.update_tsdf(raydata, depth)
        LOG_INFO(f"frame_id={frame_id}: Completed")    
    
    # check depth in reduced near and far
    n_invalid = 0 
    total_pixels = 0
    reduce_lens = []
    for frame_id in datamanager.dataset.frame_ids:
        raydatas, depths = datamanager.get_tsdf_data_with_depth(frame_id=frame_id,batch_size=4096,depth_file_dir=depth_file_dir,depth_file_fmt=depth_file_fmt,device=0)
        for raydata, depth in zip(raydatas,depths):
            ray_depth = depth / (raydata.depth_scale) # (batch, 1)
            near,far = raydata.near_far()
            # tsdf_model.update_tsdf(raydata, ray_depth)
            near,far = tsdf_model.sampler.carving_empty_space(raydata, False) # (batch,1),(batch,1)
            valid_depth =  torch.logical_and(near < ray_depth, far > ray_depth)
            # valid_depth =  near < ray_depth
            # valid_depth =  far > ray_depth
            n_invalid += torch.logical_not(valid_depth).sum().item()
            reduce_lens.append(convert_numpy(far-near))
            total_pixels += near.shape[0]
            # tsdf_model.sampler.reset()
    reduce_lens = np.concatenate(reduce_lens,axis=0)        
    LOG_INFO("invalid pixels: {}/{} = {}".format(n_invalid, total_pixels,n_invalid / total_pixels))
    LOG_INFO("mean reduced length: {}({})".format(reduce_lens.mean().item(),reduce_lens.std().item()))

if __name__ == "__main__":
    app.run(main)
