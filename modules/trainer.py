from torch.nn.parallel import DistributedDataParallel as DDP
from utils.configs import TrainConfig
from utils.writer import TensorBoardWriter
from models.model import load_model
# from models.predfined.monosdf import MonoSDFModel
from modules.datamanager import *
from modules.loss.losses import *
from modules.render import *
from modules.optimizer import *
from modules.saver import * 
from utils.logger import * 
from modules.saver import *
from modules.render import render
from utils.cv_utils.image_utils import *
from time import time 



class Trainer:
    def __init__(self,local_rank: int, world_size: int, config: TrainConfig):
        """
        TODO:
            1. set dataloader
            2. set model
            3. load/save checkpoint
            4. train model
            5. evaluate
        """
        self.config:TrainConfig = config
        self.datamanager = DataManager(config, world_size)
        self.local_rank:int = local_rank
        self.world_size:int = world_size
        self.start_step = 0
        self.num_steps = config.num_steps
        self.model = load_model()
        self.model = self.model.to(local_rank)
        
        self.optimizer = get_optimizer(config, self.model)
        decay_step = self.num_steps
        self.scheduler = get_scheduler(config,self.optimizer,decay_step)
        # self.logger = logger(config)
        self.loss = TotalLoss(config)
        self.logdir = osp.join(config.checkpoint_path, "log")
        os.makedirs(self.logdir,exist_ok=True)
        self.writer = TensorBoardWriter(self.logdir)
        if config.continue_path:
            self._load_checkpoint(config.continue_path)
        if world_size > 1: self.train_model = DDP(self.model)
        else: self.train_model = self.model
        
        if config.eval_random_sample or len(config.eval_frame_ids) == 0:
            self.eval_frame_ids = random.sample(range(0,self.datamanager.size),config.num_eval_images)
        else:
            self.eval_frame_ids = config.eval_frame_ids
            

    def train(self):
        # print start training
        self.train_model.train()
        for step in range(self.start_step,self.start_step + self.num_steps):
            loss_dict,stat_dict,meta_data =self._train_per_loop(step)
            if step % self.config.step_per_log == 0:
                self._print_terminal_log(loss_dict,stat_dict,meta_data,step)
            if self.local_rank == 0 and step % self.config.step_per_eval == 0:
                self._evaluate(step)
                self.train_model.train()
            if self.local_rank == 0 and step % self.config.step_per_save == 0:
                self._save_checkpoint(step) 
        # print end training
        LOG_INFO("Training Finished.")    
        self._save_checkpoint(self.start_step + self.num_steps)
    
    def test(self):
        self.model.eval()
        # render novel view
        height, width = self.config.image_size
        phi = height / width * 180
        novel_cam = EquirectangularCamera({
            "image_size": self.config.image_size,
            "min_phi_deg": -phi,
            "max_phi_deg": phi,
        })
        render_video(self.config, self.model,self.datamanager,0,novel_cam,video_path=osp.join(self.config.checkpoint_path,self.config.video_file))
    
    def _train_per_loop(self, step:int):
        self.optimizer.zero_grad()
        raybatch,databatch = self.datamanager.next_train_ray(step)
        raybatch.to(self.local_rank)
        databatch.to(self.local_rank)
        raw_output, render_output, meta_data = self.train_model(raybatch,step)
        total_loss, loss_dict, stat_dict = self.loss.compute_total_loss(raw_output, render_output, databatch,step)
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # write loss values
        for key in loss_dict:
            self.writer.write_scalar(f"loss/{key}",loss_dict[key],step)
        # write statistics
        # for debug
        stat_dict["lr"] = self.optimizer.param_groups[0]['lr']
        for key in stat_dict:
            self.writer.write_scalar(f"statistics/{key}",stat_dict[key],step)
        # write meta
        for key in meta_data:
            self.writer.write_scalar(f"meta/{key}",meta_data[key],step)
        return loss_dict, stat_dict, meta_data

    def _save_checkpoint(self, step:int):
        save_path = osp.join(self.config.checkpoint_path, f"{step:06d}.pt")
        save_checkpoint(save_path,step,self.model.network,self.model.sampler,self.optimizer)
        LOG_INFO(f"Complete to save checkpoint:{save_path}")

    def _load_checkpoint(self, path:str):
        checkpoint_dict = load_checkpoint(path)
        self.model.network.load_state_dict(checkpoint_dict["network"])
        self.model.sampler.load_state_dict(checkpoint_dict["sampler"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_step = checkpoint_dict["step"]
        
    def _print_terminal_log(self, loss_dict:Dict[str,float], stat_dict:Dict[str,float],meta_data:Dict[str,float],step:int):
        loss_list = [f"{key}={loss_dict[key]:04f}" for key in loss_dict]
        stat_list= [f"{key}={stat_dict[key]:04f}" for key in stat_dict]
        if len(meta_data):
            meta_list = {f"{key}={meta_data[key]:04f}" for key in meta_data}
            LOG_INFO( f"{step}/{self.start_step + self.num_steps}: " +\
                ", ".join(loss_list) + ", "+", ".join(stat_list) + ", "+", ".join(meta_list))
        else:
            LOG_INFO( f"{step}/{self.start_step + self.num_steps}: " +\
                ", ".join(loss_list) + ", "+", ".join(stat_list))
            
    def _evaluate(self, step):
        self.model.eval()
        # novel_cam = EquirectangularCamera(cam_dict={
        #     "image_size": (256,1024),
        #     "min_phi_deg": -45,
        #     "max_phi_deg": 45
        #     })
        for frame_id in self.eval_frame_ids:
            ## train view
            batchlist, image_size = self.datamanager.get_rendering_train_data(frame_id,self.config.render_batch_size,self.local_rank)
            render_output = render(batchlist,self.model, image_size)
            render_output["rgb"] = concat_images([self.datamanager.dataset.get_image_by_frame_id(frame_id),convert_image(render_output["rgb"])],vertical=False) 
            render_output["inv_depth"] = float_to_image(render_output["depth"],\
                self.config.depth_min_val,self.config.depth_max_val)
            if "normal" in render_output: 
                render_output["normal"] = normal_to_image(render_output["normal"])
            self.writer.write_render_dict(frame_id,render_output,step)
            # ## novel view
            # if frame_id != self.datamanager.dataset.frame_ids[-1]:
            #     pose,tp = interpolate_info(dataset=self.datamanager.dataset,frame_id=frame_id,t=0.5)
            # else:
            #     pose = self.datamanager.dataset.get_cam2world_by_frame_id(frame_id)
            #     tp = self.datamanager.dataset.get_timestamp_by_frame_id(frame_id)
            # pose.t += np.array([0.1,0.1,0.1])
            # novel_batchlist, novel_image_size = self.datamanager.get_rendering_data(frame_id,novel_cam,pose,tp,\
            #     self.config.render_batch_size,self.local_rank)
            # novel_view_render = render(novel_batchlist,self.model, novel_image_size)
            # novel_view_render["novel/rgb"] = convert_image(novel_view_render["rgb"])
            # novel_view_render["novel/depth"] = depth_to_image(novel_view_render["depth"],\
            #     self.config.depth_min_val,self.config.depth_max_val)
            # if "normal" in novel_view_render: 
            #     novel_view_render["novel/normal"] = normal_to_image(novel_view_render["normal"])
            # self.writer.write_render_dict(frame_id,novel_view_render,step)
    