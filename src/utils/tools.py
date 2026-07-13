import math
import collections
from itertools import repeat
import torch
import os
from torch import optim

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse
_pair = _ntuple(2, "_pair")

class LossRecorder():
    def __init__(self):
        super().__init__()
        self.loss = 0
        self.bpp_y_loss = 0
        self.bpp_z_loss = 0
        self.bpp_offset_loss = 0
        self.total_mse_loss = 0
        self.total_step = 0
        self.total_ms_ssim_loss = 0
        self.lpips_loss = 0
    def update_loss(self, loss, info):
        
        if info['bpp_y'] is not None:
            self.loss += loss
           
            self.bpp_y_loss += info['bpp_y']
            self.bpp_z_loss += info['bpp_z']
            self.bpp_offset_loss += info['bpp_offset']
            self.total_mse_loss += info['mse_loss']
            
            self.total_step += 1
        else:
            self.loss += loss
            self.mse_loss += info['mse_loss']
            self.total_step += 1
        if info['lpips_loss'] is not None:
            self.lpips_loss +=  info['lpips_loss']
            self.total_ms_ssim_loss += info['ssim_loss']
            
    def print_average_loss(self, end='\n'):
        assert self.total_step > 0
        print(f'Loss: {self.loss/self.total_step:.4f} |'
             f'mse_loss: {self.total_mse_loss/self.total_step:.6f} |'
             f'ssim_loss: {self.total_ms_ssim_loss/self.total_step:.6f} |'
              f'lpips_loss: {self.lpips_loss/self.total_step:.6f} |'
              f'bpp_y: {self.bpp_y_loss/self.total_step:.6f} |'
              f'bpp_offset: {self.bpp_offset_loss/self.total_step:.6f} |'
              f'bpp_z: {self.bpp_z_loss/self.total_step:.6f} |', end=end)
        return self.loss/self.total_step,  (f'Loss: {self.loss/self.total_step:.4f} |'
             f'mse_loss: {self.total_mse_loss/self.total_step:.6f} |'
             f'ssim_loss: {self.total_ms_ssim_loss/self.total_step:.6f} |'
              f'lpips_loss: {self.lpips_loss/self.total_step:.6f} |'
              f'bpp_y: {self.bpp_y_loss/self.total_step:.6f} |'
              f'bpp_offset: {self.bpp_offset_loss/self.total_step:.6f} |'
              f'bpp_z: {self.bpp_z_loss/self.total_step:.6f} |')
    def add_to_tensor_board(self, writer, epoch, prefix='train'):
        assert self.total_step > 0
        writer.add_scalar(f"{prefix}_loss", self.loss / self.total_step, self.total_step)
        writer.add_scalar(f"{prefix}_bpp_y_loss", self.bpp_y_loss / self.total_step, self.total_step)
        writer.add_scalar(f"{prefix}_bpp_z_loss", self.bpp_z_loss / self.total_step, self.total_step)
        writer.add_scalar(f"{prefix}_total_mse_loss", self.total_mse_loss / self.total_step, self.total_step)


def get_loss_func(loss_func, lmbda):
    def loss_me_mse(rd,scale):
        return lmbda * rd['me_mse']

    def loss_me_rdc_mse(rd,scale):
        return lmbda * rd['me_mse'] + rd['bpp_mv_y'] + rd['bpp_mv_z']

    def loss_recon_mse(rd,scale):
        return lmbda * rd['mse_loss']

    def loss_recon_rdc_mse(rd,scale):
        return lmbda * rd['mse_loss'] + rd['bpp_y'] + rd['bpp_z']

    def loss_total_rdc_mse(rd,scale):
        return lmbda * rd['mse_loss'] + rd['bpp']
    def loss_total_rdc_mse_vr(rd,scale):
        return scale*lmbda * rd['mse_loss'] + rd['bpp']
    def loss_total_rdc_ms_ssim_vr(rd,scale):
        return scale*lmbda/32 * rd['ssim_loss'] + rd['bpp']
    def loss_total_rdc_yuv_mse_vr(rd,scale):
        return scale*lmbda * rd['yuv_loss'] + rd['bpp']
    def loss_total_rdc_mse_lpips_vr(rd,scale):
        return scale*lmbda * 0.5*(rd['mse_loss']+ 0.006*rd['lpips_loss']) + rd['bpp'] 
    def loss_total_rdc_yuv_mse_lpips_vr(rd,scale):
        return scale*lmbda * 0.5*(rd['yuv_loss']+ 0.006*rd['lpips_loss']) + rd['bpp'] 
    
    

    loss_func_name = f'loss_{loss_func}'
    assert loss_func_name in locals()
    return locals()[loss_func_name],loss_func_name

def get_latest_checkpoint_path(dir_cur):
    files = os.listdir(dir_cur)
    all_best_checkpoints = []
    p_model,name,start_stage,start_epoch = None,None,None,None
    for file in files:
        if file[-4:] == '.pth':
            all_best_checkpoints.append(os.path.join(dir_cur, file))
    if len(all_best_checkpoints) > 0:
        p_model =  max(all_best_checkpoints, key=os.path.getmtime)
        name = p_model.split('/')[-1].split('.')[-2]
        start_stage = name.split('_')[-2]
        start_epoch = name.split('_')[-1]
        return p_model,name,start_stage,start_epoch
    
    return None,None,None,None,
def load_models(i_frame_net,video_net, i_frame_path, task_path,  training_strategy=None,device=None):
    model_dict = video_net.state_dict()
    load_p_checkpoints,name, current_stage,current_epoch= get_latest_checkpoint_path(task_path)
    keys = list(training_strategy.keys())
    if load_p_checkpoints is not None:
        load_checkpoint = torch.load(load_p_checkpoints, map_location=torch.device('cpu'))
        if current_stage in keys:
            total_stage_epoch = len(training_strategy[current_stage])
            if int(current_epoch) < total_stage_epoch-1:
                start_stage,start_epoch= current_stage,int(current_epoch)+1
            elif keys.index(current_stage) == len(keys)-1:
                start_stage =keys[-1]
                start_epoch = total_stage_epoch
            else:
                start_stage = keys[keys.index(current_stage) + 1]
                start_epoch = 0
            print(f"load model from: {name} | start_stage is {start_stage} start_epoch is {start_epoch}")
            load_model_para = load_checkpoint['state_dict']
            load_model_para = {k: v for k, v in load_model_para.items() if k in model_dict}
            model_dict.update(load_model_para)
            video_net.load_state_dict(model_dict, strict = False)
    elif 'finetune' in task_path:
        '''如果当前是fineune阶段且没有找到可以load的模型，则回到上层文件夹搜索pretrain 模型'''
        base_path =  task_path.split("finetune")[0]
        pretrain_path = base_path + 'pretrain/'
        load_p_checkpoints, name, _, _ = get_latest_checkpoint_path(pretrain_path)
        start_stage = keys[0]
        start_epoch = 0
        video_net.load_pretrained_weights(load_p_checkpoints)
    else:
        start_stage,start_epoch="pretrain",0
        print(f"从头训练 start_stage is {start_stage} start_epoch is {start_epoch}")
    if "finetune" in start_stage:
        i_frame_path = "./cvpr2023_image_psnr.pth.tar"
    else:
        i_frame_path = None
    print(f"i_frame_net load model from: {i_frame_path}")
    if  i_frame_path is not None:
        load_checkpoint_i = torch.load(i_frame_path, map_location=torch.device('cpu'))
        i_frame_net.load_state_dict(load_checkpoint_i,strict=False)
    video_net = video_net.to(device)
    i_frame_net = i_frame_net.to(device)
    return i_frame_net,video_net,start_stage,start_epoch


    # if use_ckpt:
    #     load_checkpoint = torch.load(load_p_checkpoints, map_location=torch.device('cpu'))
    #     total_stage_epoch = len(training_strategy[current_stage])
    #     if int(current_epoch) < total_stage_epoch-1:
    #         start_stage,start_epoch= current_stage,int(current_epoch)+1
    #     else:
    #         keys = list(training_strategy.keys())
    #         start_stage = keys[keys.index(current_stage) + 1]
    #         start_epoch = 0
    #     print(f"load model from: {name} | start_stage is {start_stage} start_epoch is {start_epoch}")
    #     load_model_para = load_checkpoint['state_dict']
    #     load_model_para = {k: v for k, v in load_model_para.items() if k in model_dict}
    #     model_dict.update(load_model_para  )
    #     video_net.load_state_dict(model_dict, strict = False)
    # elif  load_p_checkpoints is not None:
    #     video_net.load_pretrained_weights(load_p_checkpoints)
    #     total_stage_epoch = len(training_strategy[current_stage])
    #     if int(current_epoch) < total_stage_epoch-1:
    #         start_stage,start_epoch= current_stage,int(current_epoch)+1
    #     else:
    #         keys = list(training_strategy.keys())
    #         start_stage = keys[keys.index(current_stage) + 1]
    #         start_epoch = 0
    #     print(f"load model from: {name} | start_stage is {start_stage} start_epoch is {start_epoch}")
    # else:
    #     start_stage,start_epoch="residue",0
    #     print(f"从头训练 start_stage is {start_stage} start_epoch is {start_epoch}")
    # if "finetune" in start_stage:
    #     i_frame_path = "./cvpr2023_image_yuv420_psnr.pth.tar"
    # else:
    #     i_frame_path = None
    # print(f"i_frame_net load model from: {i_frame_path}")
    # if  i_frame_path is not None:
    #     load_checkpoint_i = torch.load(i_frame_path, map_location=torch.device('cpu'))
    #     i_frame_net.load_state_dict(load_checkpoint_i,strict=False)
    # video_net = video_net.to(device)
    # i_frame_net = i_frame_net.to(device)
    
    # return i_frame_net,video_net,start_stage,start_epoch

def load_models_i(i_frame_net,current_stage,device):
    if "finetune" in  current_stage:
        i_frame_path = "./cvpr2023_image_yuv420_psnr.pth.tar"
    else:
        i_frame_path = None
    i_frame_net = i_frame_net.to('cpu')
    print(f"i_frame_net load model from: {i_frame_path}")
    if  i_frame_path is not None:
        load_checkpoint_i = torch.load(i_frame_path, map_location=torch.device('cpu'))
        i_frame_net.load_state_dict(load_checkpoint_i,strict=False)
    i_frame_net = i_frame_net.to(device)
    
    return i_frame_net


def train_init(video_net,strategy,idx,logger,lmbda,train_dataloader):
    coding_mode = strategy[idx][1]
    lr = strategy[idx][0]
    train_name = []
    untrain_name = []
    #  优化器设置
    # if coding_mode =="residue":
    for name,param in video_net.named_parameters():
        if 'lpips' in name:
            param.requires_grad =False
        # elif 'flow' in name:
        #     param.requires_grad =False
            # elif 'q_scale_' in name:
            #     param.requires_grad =False
            # elif 'q_basic_' in name:
            #     param.requires_grad =False
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, video_net.parameters()) , lr=lr)
    # elif coding_mode =="all":
    #     for name,param in video_net.named_parameters():
    #         if 'optic' in name:
    #             param.requires_grad =False
    #         elif 'q_scale_' in name:
    #             param.requires_grad =False
    #         elif 'q_basic_' in name:
    #             param.requires_grad =False
    # else:
    #     for name,param in video_net.named_parameters():
    #         if 'optic' in name:
    #             param.requires_grad =False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, video_net.parameters()) , lr=lr)    
    
    for name,param in video_net.named_parameters():
        parent_name = name.split('.')[0]
        if param.requires_grad ==True:
            if parent_name not in train_name:
                train_name.append(parent_name)
        else:
            if parent_name not in untrain_name:
                untrain_name.append(parent_name)
    ## 损失函数设置    
    loss_func,loss_func_name  = get_loss_func(strategy[idx][2], lmbda)
    use_lpip = False
    use_yuv = False
    if "lpips" in loss_func_name or 'ssim' in loss_func_name:
        use_lpip = True
    elif "yuv" in loss_func_name:
        use_yuv = True
    ## 数据集设置
    train_seq = strategy[idx][3]
    train_dataloader.dataset.set_frame_num(train_seq)
    ## 信息打印
    print(f" coding_mode:  {coding_mode}|    lr:  {lr}    |")
    logger.info(f"********** coding_mode:  {coding_mode}   |   lr:  {lr}   |********************")
    logger.info(f"**********parameters information**********  ")
    logger.info(f"trained {train_name}")
    logger.info(f"fixed{untrain_name}")
    return optimizer,loss_func, train_dataloader,coding_mode,use_lpip,use_yuv