import gin
import os
import os.path as osp
from absl import flags
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, List
import shutil

gin.add_config_file_search_path('configs/')
FLAGS = flags.FLAGS

@gin.configurable()
@dataclass
class TrainConfig:
    checkpoint_path:str = ""
    dataset_normalize:bool = False
    #optimizer & scheduler
    optimizer:str = "Adam"
    lr:float = 0.0005
    multiplier_grid_lr:float = 20.
    betas:Tuple[float,float] = (0.9,0.99)  
    decay_rate:float = 0.1
    
    continue_path:str = ""
    model_config_file:str = ""
    dataset_config_file:str = ""
    
    # training configs
    num_steps:int = 50000 # iteration steps
    train_batch_size:int = 2048

    step_per_eval:int = 5000
    step_per_save:int = 5000
    step_per_log:int = 100
    
    # evaluation configs
    num_eval_images:int = 3
    eval_random_sample:bool = True
    eval_frame_ids:List[int] = field(default_factory= lambda: [])
    render_batch_size:int = 2048

    # loss configs
    loss_weight: Dict[str,float] = field(default_factory= lambda:
        {
            "coarse_rgb": 0.5,
            "rgb": 1.,
        })
    
    # render   
    depth_min_val:float =  0.1
    depth_max_val:float =  2.5
    
    # final test
    interpolation_times:int = 1
    translation_offset:Tuple[int] = (0.,0.,0.)
    
    use_novel_cam:bool = False
    image_size:Tuple[int] = (256,1024)
    
    video_file:str = "render_video.mp4"
    vertical_concat:bool = False
    fps:int = 5

    use_tsdf:bool = False

@gin.configurable()
@dataclass
class RenderConfig:
    checkpoint_path:str = ""
    checkpoint_step:int = 0
    model_config_path:str = ""

    render_batch_size:int = 4096
    interpolation_times:int = 1
    translation_offset:Tuple[int] = (0.,0.,0.)
    
    use_novel_cam:bool = False
    image_size:Tuple[int] = (256,1024)

    depth_min_val:float =  0.1
    depth_max_val:float =  2.5

    video_file:str = "render_video.mp4"
    vertical_concat:bool = False
    fps:int = 5

def load_train_config(config_path:str=None,save_config:bool=True):
    if config_path: gin.parse_config_file(config_path) 
    elif FLAGS.gin_configs: gin.parse_config_file(FLAGS.gin_configs)

    config = TrainConfig()
    # parse model_config_file
    gin.parse_config_file(config.model_config_file)
    gin.parse_config_file(config.dataset_config_file)
    if save_config:save_config_gin(config)
    return config

def load_render_config(config_path:str=None):
    if config_path: gin.parse_config_file(config_path) 
    elif FLAGS.gin_render_configs: gin.parse_config_file(FLAGS.gin_render_configs)
    config = RenderConfig()
    if config.model_config_path == "":
        model_config_path = osp.join(config.checkpoint_path,"model_config.gin")
    else: model_config_path = config.model_config_path
    gin.parse_config_file(model_config_path)
    
    return config

def save_config_gin(config: TrainConfig):
    os.makedirs(config.checkpoint_path, exist_ok=True)
    config_dict = asdict(config)
    with open(osp.join(config.checkpoint_path, "config.gin"),'w') as f:
        f.write("# Parameters for Config:\n")
        f.write("# ==============================================================================\n")
        for key in config_dict:
            f.write(f'Config.{key} = {config_dict[key]}\n') # train config
    # save model config
    shutil.copyfile(config.model_config_file, osp.join(config.checkpoint_path,"model_config.gin"))
    # sace dataset config
    shutil.copyfile(config.dataset_config_file, osp.join(config.checkpoint_path,"dataset_config.gin"))


if __name__ == "__main__":
    config = TrainConfig()
    print(config.loss_weight)