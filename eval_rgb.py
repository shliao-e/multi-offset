import argparse

import numpy as np
import time
import torch
import os
from torch.utils.data import DataLoader
from src.datasets.test_datasets import CTS,CTS_yuv
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from src.models.video_simplergb import Multi_DMC
from src2.models.image_model import IntraNoAR
# from src.models.video_net_dmc import DMC
import csv
import torch.nn.functional as F
from tqdm import tqdm
# from src.models.priors import model_architectures as architectures
from src.utils.common import str2bool
from src2.utils.stream_helper import get_padding_size
from src.transforms.functional import yuv_444_to_420,ycbcr420_to_rgb
import lpips

parser = argparse.ArgumentParser(description="Example training script")


parser.add_argument("--p_model", type=str, default='Argb_checkpoints/finetune/')
parser.add_argument('--model_name', type=str, default="finetune2_7", help='models_name')
parser.add_argument("--i_model", type=str, default='./cvpr2023_image_psnr.pth.tar')
parser.add_argument('--gop_size', type=int,default=40)
parser.add_argument('--lambda', dest='lmbda', type=int, default=4)

parser.add_argument('--cuda_id', type=int, default= 0, help='Use cuda')
parser.add_argument('--save_dir', type=str, default="./", help='Path to save models')
# parser.add_argument('--test_class', type=str, default="ClassB", help='Path to save models')
parser.add_argument('--test_class', type=str, default="huawei-test2", help='Path to save models')
args = parser.parse_args()



def write_yuv420_sequence(y_list, u_list, v_list, save_path):
    with open(save_path, "wb") as f:
        for y, u, v in zip(y_list, u_list, v_list):
            f.write(y.tobytes())
            f.write(u.tobytes())
            f.write(v.tobytes())


def test_one_epoch(video_net, image_net, test_dataloader,  epoch, device = None,coding_mode = None,writer = None,loss_fn =None):
    video_net.eval()
    video_net.to(device)
    image_net.to(device)
    write_dir =  f'save_seq/lambda{args.lmbda}'
    img_save_dir = os.path.join(write_dir, "recon_rgb",args.model_name)
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(write_dir, exist_ok=True)
    # 创建统计文件保存目录
    stats_dir = os.path.join(write_dir, "seq_stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # 为每个epoch创建统计文件
    seq_stats_file = os.path.join(stats_dir, f"results_{args.test_class}.txt")
    header = "Seq_Index\tPSNR_Avg\tLPIPS\tBPP\trate\tFrame_Count"
    with open(seq_stats_file, 'w', encoding='utf-8') as f:
        f.write(header + "\n")
        print(f"已创建序列统计文件: {seq_stats_file}")
        print(header)
    # 码率换算参数
    FRAME_RATE = 10  # 帧率 10fps
    WIDTH = 960      # 宽度 960
    HEIGHT = 540     # 高度 540
    TOTAL_PIXELS = HEIGHT * WIDTH  # 每帧总像素数
    def bpp_to_kbps(bpp, fps=FRAME_RATE, total_pixels=TOTAL_PIXELS):
        """将bpp转换为kbps
        bpp: bits per pixel
        fps: 帧率
        total_pixels: 每帧总像素数
        """
        return (bpp * total_pixels * fps) / 1000
    sum_psnr = 0
    sum_lpips = 0
    sum_psnr_y = 0
    sum_psnr_u = 0
    sum_psnr_v = 0
    sum_mse =0
    sum_bpp = 0
    sum_kbps = 0 
    cnt = 0
    for b, frames, in enumerate(test_dataloader):
        seq_lpips = 0
        seq_psnr_avg = 0
        seq_mse = 0
        seq_bpp = 0
        seq_cnt = 0
        seq_kbps = 0
        test_frames = frames['frames'].to(device)
        batch_size, frame_length, _, h, w = test_frames.shape

        recon_y_seq = []

        
        for frame_idx in tqdm(range(frame_length)):
        # for frame_idx in tqdm(range(40)):
            with torch.no_grad():
                cur_frame = test_frames[:, frame_idx, :, :]
                
                padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w, 64)
                cur_frame_pad = torch.nn.functional.pad(
                cur_frame,
                (padding_l, padding_r, padding_t, padding_b),
                mode="replicate",)
                # if (frame_idx % args.gop_size == 0 and frame_idx != 80) or frame_idx ==82:
                if frame_idx % args.gop_size == 0:
                    ref_y =None
                    ref_feature = None
                    i_q_index = {'1024':3, '128':2, '64':1, '16': 0, '8': 0, '4': 0, '2': 0, '1': 0}
                    q_index = i_q_index[str(args.lmbda)]
                    ##执行I帧分辨率下采样
                    down_factor = 1/2
                    cur_frame_resize = F.interpolate(cur_frame,scale_factor=down_factor,mode="bilinear")
                    padding_l_d, padding_r_d, padding_t_d, padding_b_d = get_padding_size(int(h*down_factor), int(w*down_factor), 64)
                    cur_frame_resize_pad = torch.nn.functional.pad(
                            cur_frame_resize,
                            (padding_l_d, padding_r_d, padding_t_d, padding_b_d),
                            mode="replicate",)
                    result = image_net(cur_frame_resize_pad,True,q_index)
                    ref_frame = result['x_hat'].detach()
                     #************指标计算**************
                    recon_frame = F.pad(ref_frame, (-padding_l_d, -padding_r_d, -padding_t_d, -padding_b_d))
                    recon_frame =  F.interpolate(recon_frame ,scale_factor=1/down_factor,mode="bilinear")
                    recon_frame = recon_frame.clamp_(0.,1.)
                    save_image(recon_frame, os.path.join(img_save_dir, f"seq{b}_frame{frame_idx}.png"))

                    ref_frame = torch.nn.functional.pad(
                                                        recon_frame,
                                                        (padding_l, padding_r, padding_t, padding_b),
                                                        mode="replicate",)

                    # 计算Y分量的MSE和PSNR
                    intra_mse = F.mse_loss(recon_frame, cur_frame).item()
                    intra_psnr = 10 * (np.log(1 * 1 / intra_mse) / np.log(10))

                    #计算lpips
                    lpips_loss = loss_fn(recon_frame,cur_frame).mean().item()
                    # 计算平均PSNR（可以加权，这里简单平均）
                    #指标统计
                    seq_lpips += lpips_loss
                    bits = result['bit'].item()
                    bpp =bits / (h*w)
                    seq_bpp += bpp
                    seq_psnr_avg += intra_psnr
                    seq_mse += intra_mse
                    seq_cnt += 1
                    writer.writerow([frame_idx,intra_psnr,bpp])
                    sum_lpips+=lpips_loss
                    sum_bpp += bpp
                    sum_mse += intra_mse
                    sum_psnr += intra_psnr
                    cnt +=1
                    dpb = {
                        'ref_frame': ref_frame,
                        'ref_feature': ref_feature,
                        'ref_y':ref_y,
                        'ref_feature_L': None,
                        } 
                    continue
                idx_ = frame_idx % 8
                result = video_net.forward_one_frame(cur_frame_pad,dpb,idx_,True)               
                if frame_idx >0 :
                    ref_feature = result['dpb']['ref_feature'].detach()
                    ref_y = result['dpb']['ref_y'].detach()
                    ref_frame = result['dpb']["ref_frame"].detach()
                    dpb = {
                        'ref_frame': ref_frame,
                        'ref_feature': ref_feature,
                        'ref_y':ref_y,
                        }   
                # if frame_idx % 20 ==0:
                #     dpb = {
                #     'ref_frame': ref_frame,
                #         'ref_feature': ref_feature,
                #         'ref_y':None,
                #         'ref_feature_L': ref_feature_L,
                #     } 
                
               #************指标计算**************
                recon_frame = F.pad(ref_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
                inter_mse =  F.mse_loss(recon_frame, cur_frame).item()
                inter_psnr =10 * (np.log(1 * 1 / inter_mse) / np.log(10))
                lpips_loss = loss_fn(recon_frame,cur_frame).mean().item()
               
                save_image(recon_frame, os.path.join(img_save_dir, f"seq{b}_frame{frame_idx}.png"))
                # 计算平均PSNR（可以加权，这里简单平均）
                seq_lpips+= lpips_loss
                # inter_psnr = (6*psnr_y + psnr_u + psnr_v) / 8
                bpp = result['bpp'].item()
                seq_bpp += bpp
                seq_psnr_avg += inter_psnr
                seq_mse += inter_mse
                seq_cnt += 1
                writer.writerow([frame_idx,inter_psnr,bpp])
                sum_lpips+=lpips_loss
                sum_bpp += bpp
                sum_mse += inter_mse
                sum_psnr += inter_psnr
                cnt +=1
        seq_kbps = bpp_to_kbps(seq_bpp/seq_cnt)
        print(f" avg_psnr:{seq_psnr_avg / seq_cnt:.4f} " 
        f"seq_lpips: {seq_lpips/seq_cnt:.4f}"   f"avg_BPP: {seq_bpp/seq_cnt:.4f}"
        f"rate: {seq_kbps:.4f}"
          )
        with open(seq_stats_file, 'a', encoding='utf-8') as f:
            f.write(f"{b}\t")  # 序列索引
            f.write(f"{seq_psnr_avg / seq_cnt:.4f}\t")  # PSNR_Avg
            f.write(f"{seq_lpips / seq_cnt:.4f}\t")  # LPIPS
            f.write(f"{seq_bpp / seq_cnt:.4f}\t")  # BPP
            f.write(f"{seq_kbps:.4f}\t")  # rate
            f.write(f"{seq_cnt}\n")  # 帧数

    sum_kbps = bpp_to_kbps(sum_bpp/cnt)
    with open(seq_stats_file, 'a', encoding='utf-8') as f:
        f.write("\n")  # 空行分隔
        f.write("Overall_Average\t")
        f.write(f"{sum_psnr / cnt:.4f}\t")
        f.write(f"{sum_bpp / cnt:.4f}\t")
        f.write(f"{sum_kbps:.4f}\t")
        f.write(f"{cnt}\n")
    return sum_psnr_y / cnt,sum_psnr_u / cnt,sum_psnr_v / cnt,sum_psnr / cnt, sum_bpp/cnt, sum_mse / cnt ,sum_lpips/cnt

def eval():
    file_dir = './data.csv'
    f = open(file_dir, 'w', encoding= 'utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["num","PSNR","BPP","MSE",])
    # test_dataset = CTS_yuv("/home/tione/notebook/datasets",f"{args.test_class}",output_rgb=True)
    test_dataset = CTS("/home/tione/notebook/datasets",f"{args.test_class}",output_yuv=False)
    test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=1,shuffle=False)
    writer = SummaryWriter(f"{args.save_dir}/plot")

    device = torch.device(f"cuda:{args.cuda_id}")
    video_net = Multi_DMC()
    # video_net = DMC()
    image_net = IntraNoAR()
    # msg = video_net.cal_macs()
    # print("*"*10,'复杂度',"*"*10,"/n",msg)
    p_model_path = os.path.join(args.p_model,str(args.lmbda), args.model_name)+".pth"
    load_checkpoint = torch.load(p_model_path, map_location=torch.device('cpu'),weights_only=True)
    load_checkpoint_i = torch.load(args.i_model, map_location=torch.device('cpu'),weights_only=True)
    print(f'***************load from : {p_model_path}*************')
    load_model_para = load_checkpoint['state_dict']
    video_net.load_state_dict(load_model_para, strict = False)
    # video_net.load_pretrained_weights( p_model_path)
    # video_net.update(force=True)
    image_net.load_state_dict(load_checkpoint_i,strict=False)
    loss_fn = lpips.LPIPS(net='alex').to(device)
    eval_psnr_y, eval_psnr_u, eval_psnr_v,eval_psnr, eval_bpp , eval_mse_loss,eval_lpips = test_one_epoch(video_net,image_net,test_dataloader,1,device,"all", csv_writer,loss_fn)
    eval_loss = eval_bpp + args.lmbda *eval_mse_loss
    print(
              f" loss is {eval_loss}"
              f" eval_mse_loss is {eval_mse_loss}"
               f" Y_psnr is : {eval_psnr_y}"   f" U_psnr is : {eval_psnr_u}"   f" V_psnr is : {eval_psnr_v}"
               f" Lpips is : {eval_lpips}"
               f" psnr is : {eval_psnr}"
                f" BPP is : {eval_bpp}")
    
    
if __name__ == "__main__":
    eval()