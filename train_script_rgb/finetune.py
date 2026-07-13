import argparse
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
import random
import numpy as np
import torch

from torch.utils.data import DataLoader


from src.datasets.video_dataset import VimeoFolder
from src.datasets.bvidvc_dataset import bvidvcFolder
from src.models.video_simplergb import Multi_DMC
from src2.models.image_model import IntraNoAR
from src.utils.logger import get_logger,Logger
from train_one_epoch import train_one_epoch
from src.utils.common import str2bool
from src.utils.tools import load_models,train_init,load_models_i

parser = argparse.ArgumentParser(description="Example training script")
parser.add_argument('--cuda_id', type=int, default= 1, help='Use cuda')
parser.add_argument('--lmbda', type=int, default=4)
parser.add_argument('--logger_name', type=str, default="train")
parser.add_argument("--load_new_optic", type=int, default=0)
parser.add_argument('-n', '--num_workers', type=int, default=1,
                    help='Dataloaders worker per trainer')
parser.add_argument("--i_model", type=str, default='./')
parser.add_argument('--save_dir', type=str, default="./Argb_checkpoints/finetune/", help='Path to save models')
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--train_patch_size', type=int, nargs=2, default=(256, 256))
parser.add_argument('--train_frame_num', type=int, default=7)
parser.add_argument('--train_crop_method', type=str, default='random')
parser.add_argument('--train_frame_selection', type=str, default='random')
parser.add_argument('--train_max_frame_distance', type=int, default=6)
parser.add_argument('--train_random_flip', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--train_max_zoom_factor', type=float, default=1.0)
parser.add_argument('--train_min_zoom_factor', type=float, default=1.0)
parser.add_argument('--training_scheduling', type=str, default=None,
                    help='How to schedule the training strategy, support normal and fast')
parser.add_argument('--use_ckpt', type=int, default=1)
parser.add_argument("--Pyuv420", type=str2bool,default=True)
parser.add_argument("--Iyuv420", type=str2bool,default=False)
args = parser.parse_args() 

  
training_strategy = {
    'finetune': [[ 1e-5,  "all2",     "total_rdc_mse_vr",   7]]     * 3  +\
                [[ 5e-6,  "all2",     "total_rdc_mse_vr",   7]]     * 3  +\
                [[ 1e-6,  "all2",     "total_rdc_mse_vr",   7]]     * 3 ,
    'finetune2':[[ 5e-6,  "all3",     "total_rdc_mse_vr",   16]]    * 3  +\
                [[ 3e-6,  "all3",     "total_rdc_mse_vr",   24]]    * 3  +\
                [[ 1e-6,  "all3",     "total_rdc_mse_vr",   32]]    * 2  ,
    'finetune3':[[ 5e-6,  "all3",     "total_rdc_mse_lpips_vr",   32]]    * 3  +\
                [[ 1e-6,  "all3",     "total_rdc_ms_ssim_vr",   32]]    * 3  +\
                [[ 1e-7,  "all3",     "total_rdc_ms_ssim_vr",   32]]    * 3  ,
    # 'finetune4':[[ 5e-6,  "all3",     "total_rdc_mse_lpips_vr",   32]]    * 3  +\
    #             [[ 1e-6,  "all3",     "total_rdc_mse_lpips_vr",   32]]    * 3  +\
    #             [[ 1e-7,  "all3",     "total_rdc_mse_lpips_vr",   32]]    * 3  ,            
}
stage_name = list(training_strategy.keys())
def train():
    device = torch.device(f"cuda:{args.cuda_id}")
    dataset1 = VimeoFolder("/home/tione/notebook/datasets/Vimeo/sequences/",
                                    "./configs/vimeo.npy",
                                    args.train_patch_size[0], args.train_patch_size[1],
                                    args.train_frame_num,
                                    crop_method=args.train_crop_method,
                                    frame_selection=args.train_frame_selection,
                                    max_frame_distance=args.train_max_frame_distance,
                                    max_zoom_factor=args.train_max_zoom_factor,
                                    min_zoom_factor=args.train_min_zoom_factor,
                                    random_flip=args.train_random_flip,
                                    Pyuv444=False)
    dataset2 = bvidvcFolder("/home/tione/notebook/datasets/bvi-dvc/sequences/",
                                    "./configs/bvidvc.npy",
                                    args.train_patch_size[0], args.train_patch_size[1],
                                    args.train_frame_num,
                                    crop_method=args.train_crop_method,
                                    frame_selection=args.train_frame_selection,
                                    max_frame_distance=args.train_max_frame_distance,
                                    max_zoom_factor=args.train_max_zoom_factor,
                                    min_zoom_factor=args.train_min_zoom_factor,
                                    random_flip=args.train_random_flip,
                                    Pyuv444=False)
    task_path = f"{args.save_dir}"+f"{args.lmbda}/"
    logger_path = task_path + "logfiles"  
    if not os.path.exists(task_path):
        os.makedirs(task_path)  
    video_net =Multi_DMC()
    i_frame_net =IntraNoAR()
    msg = video_net.cal_macs()
    print("*"*10,'复杂度',"*"*10,"/n",msg)
    print("*"*10,'finetune stage begin',"*"*10,"/n")
    i_frame_net, video_net, start_stage,start_epoch= \
        load_models(i_frame_net,video_net,args.i_model,task_path,training_strategy,device=device)
    if args.training_scheduling is not None:
        start_stage = args.training_scheduling.split('_')[-2]
        start_epoch = args.training_scheduling.split('_')[-1]
    current_epoch = start_epoch
    for i in range(stage_name.index(start_stage),len(stage_name)):
        if i > 0:
            train_dataset = dataset2
        else:
            train_dataset = dataset1
        train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size,\
        num_workers=args.num_workers,shuffle=True,pin_memory=True,)
        current_stage = stage_name[i]
        lmbda = args.lmbda
        logger_name = current_stage + f"_{current_epoch}"
        Logger(root_path = logger_path, logger_name = logger_name)
        logger = get_logger()
        strategy = training_strategy[current_stage]
        total_epoch = len(strategy)
        train_mode = "video"
        logger.info(f"********** Current strategy: {current_stage}   |   Start_epoch: {start_epoch}   |   train_mode: {train_mode}   |**********")
        for epoch in range(current_epoch,total_epoch):
            video_net.train()
            idx = epoch
            optimizer,loss_func, train_dataloader,coding_mode,use_lpip,use_yuv = \
                        train_init(video_net,strategy,idx,logger,float(lmbda),train_dataloader)
            logger.info(f"********** strategy information details ********** ")  
            logger.info(f"strategy information details: {strategy}")    
            train_one_epoch(video_net,i_frame_net,idx,lmbda,\
                            loss_func,train_dataloader,optimizer,\
                            coding_mode,train_mode,logger = logger,use_lpip = use_lpip, use_yuv = use_yuv)
            ## 保存模型
            checkpoint_name = f"{current_stage}_{epoch}.pth"
            save_path = f"{task_path}/{checkpoint_name}"
            save_dict = {
            "epoch": epoch,
            "state_dict": video_net.state_dict()
            }
            torch.save(save_dict,save_path)
        current_epoch = 0
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train()
