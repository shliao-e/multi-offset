import os
import random
import time
import numpy as np
import torch
from src.utils.metrics import calc_psnr
from src.utils.common import str2bool,get_rd_info,get_loss_info
from src.utils.tools import  LossRecorder
def train_one_epoch(video_net, i_frame_net, epoch,lmbda, loss_func,train_dataloader, optimizer,coding_mode,train_mode,logger = None,use_lpip = False,use_yuv = False):
    # i_frame_net.eval()
    device = next(video_net.parameters()).device
    total_step = 0
    ################## 损失函数设定 ################
    train_cascaded_loss = False
    if train_mode == "video":
        train_cascaded_loss = True
    logger.info(f'**********train for {train_mode} mode**********' )
    loss_record = LossRecorder()
    ################## 训练数据指定 ################
    train_seq_length = train_dataloader.dataset.get_frame_num()
    print(f"Train for {train_seq_length} frames")
    #################buffer 初始化 ##################
    scale_lst = [1.6,0.4,0.6,0.4]
    start_time = time.time()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        frame_nums = d.shape[1]//3
        ref_y =None
        ref_frame = d[:, 0:3, :, :]
        ref_feature = None
        ref_feature_L = None
        if coding_mode == 'all2':
            i_q_index = {'1024':3, '256':2,'128':1, '64':0, "32": 0,'16':0,'8':0, '4': 0, '2': 0,'1':0}
            index = str(lmbda)
            with torch.no_grad():
                result = i_frame_net(d[:, 0:3, :, :],True,i_q_index[index])
                ref_frame = result['x_hat'].detach()
        elif coding_mode == 'all3' :
            i_q_index ={'1024':3, '256':2,'128':1, '64':0,"32": 0, '16':0,'8':0, '4': 0, '2': 0,'1':0}
            index = str(lmbda)
            with torch.no_grad():
                result = i_frame_net(d[:, 0:3, :, :],True,i_q_index[index])
                ref_frame = result['x_hat'].detach()
                pre_num = random.randint(0, 8) * 4
                dpb = {
                        'ref_frame':ref_frame,
                        'ref_y':ref_y,
                        'ref_feature': None,
                        'ref_feature_L':None
                       } 
                for p in range(1,pre_num):
                    cur_frame = d[:, 0:3, :, :]
                    idx_ = p % 4
                    result = video_net.forward_one_frame(cur_frame,dpb,idx_,use_lip = False)
                    ref_frame = result['dpb']['ref_frame']
                    ref_feature = result['dpb']['ref_feature']
                    ref_y = result['dpb']['ref_y']
                    ref_feature_L = result['dpb']["ref_feature_L"]
                    dpb = {
                        'ref_frame':ref_frame,
                        'ref_y':ref_y,
                        'ref_feature': ref_feature,
                        'ref_feature_L': ref_feature_L,
                        }
        ######### buffer数据添加 #########

        dpb = {
            'ref_frame': ref_frame,
            'ref_y':ref_y,
            'ref_feature': ref_feature,
            'ref_feature_L': ref_feature_L,
        } 
        sum_loss = 0
        sum_bpp = 0 
        if train_seq_length >60:
            if coding_mode == "all3":
                for frame_idx in range(1,frame_nums):
                    cur_frame = d[:, frame_idx*3:(frame_idx+1)*3, :, :]
                    idx_ = frame_idx % 4
                    result = video_net.forward_one_frame(cur_frame,dpb,idx_,use_lpip,use_yuv)
                    ref_feature = result['dpb']['ref_feature']
                    ref_y = result['dpb']['ref_y']
                    ref_frame = result['dpb']['ref_frame']
                    ref_feature_L = result['dpb']["ref_feature_L"]
                    dpb = {
                            'ref_frame':ref_frame,
                            'ref_y':ref_y,
                            'ref_feature': ref_feature,
                            'ref_feature_L': ref_feature_L,
                        }                         
                    rd_info = get_rd_info(result)
                    loss_info = get_loss_info(rd_info)
                    loss = loss_func(rd_info,scale_lst[idx_])
                    sum_loss+=loss
                    loss_record.update_loss(loss.item(), loss_info)
                    if frame_idx %32==31 :
                        if frame_idx<32:
                            sum_loss = sum_loss / 31 
                        else:
                            sum_loss = sum_loss / 32
                        optimizer.zero_grad(set_to_none=True)
                        sum_loss.backward()
                        torch.nn.utils.clip_grad_norm_(video_net.parameters(), 1.0)
                        optimizer.step()
                        sum_loss = 0
                        ref_frame = ref_frame.detach()
                        ref_y = ref_y.detach()
                        ref_feature = ref_feature.detach()
                        ref_feature_L =  ref_feature_L.detach()
                        dpb = {
                        'ref_frame':ref_frame,
                        'ref_feature': ref_feature,
                        'ref_y':ref_y,
                        'ref_feature_L': ref_feature_L,
                        } 
        else:   
                    
            if train_cascaded_loss:
                optimizer.zero_grad(set_to_none=True)
                for frame_idx in range(1, frame_nums):
                    cur_frame = d[:, frame_idx*3:(frame_idx+1)*3, :, :]
                    idx_ = frame_idx % 4
                    result = video_net.forward_one_frame(cur_frame,dpb,idx_,use_lpip)
                    ref_frame = result['dpb']['ref_frame']
                    ref_feature = result['dpb']['ref_feature']
                    ref_y = result['dpb']['ref_y']
                    ref_feature_L = result['dpb']["ref_feature_L"]
                    dpb = {
                        'ref_frame':ref_frame,
                        'ref_feature': ref_feature,
                        'ref_y':ref_y,
                        'ref_feature_L': ref_feature_L,
                        }      
                    rd_info = get_rd_info(result)
                    loss_info = get_loss_info(rd_info)
                    loss = loss_func(rd_info,scale_lst[idx_])
                    sum_loss+=loss
                    loss_record.update_loss(loss.item(), loss_info)# info 需要包含所有bpp ,以及warp_loss, total_loss
                sum_loss = sum_loss / (frame_nums-1)
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(video_net.parameters(), 1.0)
                optimizer.step()
            else:

                for frame_idx in range(1, frame_nums):
                    cur_frame = d[:, frame_idx*3:(frame_idx+1)*3, :, :]
                    optimizer.zero_grad(set_to_none=True)
                    
                    idx_ = frame_idx % 4
                    if coding_mode=="vr":
                        idx_ = random.randint(0,3)
                    result = video_net.forward_one_frame(cur_frame,dpb,idx_,use_lpip)
                    if frame_idx > 0 :
                        ref_feature = result['dpb']['ref_feature'].detach()
                        ref_y = result['dpb']['ref_y'].detach()
                        ref_frame = result['dpb']['ref_frame'].detach()
                        ref_feature_L = result['dpb']["ref_feature_L"].detach()
                    dpb = {
                        'ref_frame':ref_frame,
                        'ref_feature': ref_feature,
                        'ref_y':ref_y,
                        'ref_feature_L': ref_feature_L,
                        }  

                    rd_info = get_rd_info(result)
                    loss_info = get_loss_info(rd_info)
                    loss = loss_func(rd_info,scale_lst[idx_])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(video_net.parameters(), 1.0)
                    optimizer.step()
                    loss_record.update_loss(loss.item(), loss_info)# info 需要包含所有bpp ,以及warp_loss, total_loss

        if i % 50 == 0 :
            curr_loss = LossRecorder()
            curr_loss.update_loss(loss.item(), loss_info)
            avg_loss = loss_record.loss/loss_record.total_step
            print(
                f"total_step {total_step+loss_record.total_step} "
                f"time: {(time.time()-start_time):.2f} train epoch {epoch}:["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f"({100. * i / len(train_dataloader):.0f}%)]"
                f"Avg_loss:  {avg_loss}",end=''
                )
            prt_loss,_ = curr_loss.print_average_loss(end='')
            avg_loss, avg_prt = loss_record.print_average_loss(end='')
             
            logger.info( 
                f"time: {(time.time()-start_time):.2f} train epoch {epoch}:["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f"({100. * i / len(train_dataloader):.0f}%)]" f"train lambda:  {lmbda}    " f"Avg_loss:  {avg_loss}" f" avg_detail: {avg_prt}"  f"   Loss: {prt_loss}  " 
                )
            print(f"lr1: {optimizer.param_groups[0]['lr']:.6f}")
            start_time = time.time()


