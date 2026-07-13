import time
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_state_dict
from .common_model import CompressionModel
from .video_net import  ResBlock
from .layers import  subpel_conv1x1, DepthConvBlock, ResidualBlockUpsample,conv3x3,partial,ResidualBlockUpsample
from ..layers.layers import  ResidualBlockWithStride2,SubpelConv2x
from torch.autograd import Function
from ..transforms.functional import ycbcr2rgb,yuv_444_to_420,ycbcr420_to_rgb
import lpips
from pytorch_msssim import ms_ssim

g_ch_src_d = 3 * 8 * 8
g_ch_recon = 192
g_ch_y = 128
g_ch_z = 128
g_ch_d = 128
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _pair

class Local_Align(nn.Module):
    '''使用紧凑特征生成多通道 Offsets 并进行局部对齐'''
    
    def __init__(self, c_in=g_ch_y, c_out=g_ch_y, groups=4, ks=2, stride=1, padding=1, dilation=1):
        super(Local_Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = _pair(ks)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        
        # 对齐后的特征融合权重
        self.weight = nn.Parameter(torch.Tensor(c_out, c_in, *self.ks))
        self.bias = nn.Parameter(torch.Tensor(c_out))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            

    def get_grid(self, flow, w, h):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=flow.device),
            torch.arange(w, device=flow.device),
            indexing='ij'
        )
        u = flow[:, 0] # dx
        v = flow[:, 1] # dy
        x = grid_x.unsqueeze(0).float() + u
        y = grid_y.unsqueeze(0).float() + v
        x = 2 * (x / (w - 1) - 0.5)
        y = 2 * (y / (h - 1) - 0.5)
        return torch.stack((x, y), dim=3)

    def forward(self, input_feat, offset):
        b, c, h, w = input_feat.shape
        channel_per_group = c // self.groups

        expanded_input = input_feat.repeat(1, self.ks[0] * self.ks[1], 1, 1)
        expanded_input = expanded_input.reshape(b * self.groups * self.ks[0] * self.ks[1], channel_per_group, h, w)
        offset = offset.reshape(-1, 2, h, w)
        aligned_fea = F.grid_sample(expanded_input, self.get_grid(offset, w, h), 
                                    mode='bilinear', align_corners=True)
        aligned_fea = aligned_fea.reshape(b, -1, h, w)
        out = F.conv2d(aligned_fea, self.weight.view(self.c_out, self.c_in * self.ks[0] * self.ks[1], 1, 1), 
                       bias=self.bias, stride=self.stride, dilation=self.dilation)
        return out


class Offsetenc(nn.Module):
    def __init__(self, c=192):
        super().__init__()

        self.fuse = nn.Sequential(
            DepthConvBlock(g_ch_y*2, g_ch_y),
            DepthConvBlock(g_ch_y, g_ch_y//2),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(g_ch_y//2, g_ch_y//2, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(g_ch_y//2, g_ch_y, 3, 2, 1)   # dx, dy
        )

    def forward(self, cur, ref):
        x = torch.cat([cur, ref], dim=1)
        feat = self.fuse(x)
        offset = self.offset_head(feat)
        return offset
    

class FeatureExtractor(nn.Module):
    def __init__(self, input_channel, channel,inplace=False):
        super().__init__()
        self.layer1 = nn.Sequential(DepthConvBlock(input_channel, channel, inplace=inplace),
                                    DepthConvBlock(channel, channel, inplace=inplace),
                                    )
        self.layer2 = nn.Sequential(DepthConvBlock(channel, channel, inplace=inplace),
                                    # DepthConvBlock(channel, channel, inplace=inplace),
                                    # DepthConvBlock(channel, channel, inplace=inplace),
                                    # DepthConvBlock(channel, channel, inplace=inplace),
                                    )
        # self.adaptor = nn.Conv2d(channel,g_ch_y,3,2,1)
        self.adaptor =nn.ModuleList([
                                    nn.Conv2d(channel,g_ch_y,3,2,1)
                                         for _ in range (4)])  
    def forward(self, feature,quant_step=None,frame_idx=0):
        layer1 = self.layer1(feature)
        out1 = layer1 * quant_step
        out1= self.adaptor[frame_idx](out1)
        out = self.layer2(layer1)
        return out, out1

class encoder(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.layer1 = nn.Sequential( DepthConvBlock(input_channel*3, input_channel, inplace=inplace),
                                    DepthConvBlock(input_channel, input_channel, inplace=inplace),
                                    DepthConvBlock(input_channel, input_channel, inplace=inplace),
                                    DepthConvBlock(input_channel, input_channel, inplace=inplace),
                                    )
        self.layer_2 = DepthConvBlock(input_channel, input_channel, inplace=inplace)
        self.layer_3 = DepthConvBlock(input_channel, input_channel, inplace=inplace)
        self.adaptor = nn.Conv2d(input_channel,channel,3, stride=2, padding=1)
        

    def forward(self, x, context1,context2, quant_step):
        out = self.conv1(x)
        out = self.layer1(torch.cat((out, context1,context2), dim=1))
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out * quant_step
        return self.adaptor(out)

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            # DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)
    def forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature
    
class ReconGeneration(nn.Module):
    def __init__(self,  inplace=False):
        super().__init__()
        self.layer1= nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            # DepthConvBlock(g_ch_recon, g_ch_recon),
            # DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.recon = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x,quant_step):
        out = self.layer1(x)
        out = out * quant_step
        out = self.recon(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out

class Feature_adaptorX(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, g_ch_d//4, 3, stride=2, padding=1)
        self.res_block1 = ResBlock(g_ch_d//4, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_d//4, g_ch_d//2, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_d//2, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_d//2, g_ch_d, 3, stride=2, padding=1)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)
        
        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        return layer3

class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='laplace', z_channel=g_ch_z, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)
        self.Feature_adaptor_X = Feature_adaptorX()
        # self.Multi_fusion = Multi_scale_ref_fusion()
        self.Feature_adaptor_I = nn.Sequential(
            DepthConvBlock(g_ch_src_d, g_ch_d, inplace=inplace),
            DepthConvBlock(g_ch_d, g_ch_d, inplace=inplace),
        )
        self.encoder = encoder(g_ch_d, g_ch_y, inplace=inplace)
        self.decoder = decoder()
        self.Offsetnet = nn.Sequential(
            DepthConvBlock(g_ch_y*2, g_ch_y),
            DepthConvBlock(g_ch_y, g_ch_y),
            DepthConvBlock(g_ch_y, g_ch_y),
        )

        self.offset_enc = nn.Sequential(
            nn.Conv2d(g_ch_y, g_ch_y, 3, 2, 1),
            DepthConvBlock(g_ch_y, g_ch_y),
            DepthConvBlock(g_ch_y, g_ch_y),
            DepthConvBlock(g_ch_y, g_ch_y),
            nn.Conv2d(g_ch_y, g_ch_y, 3, 2, 1),
        )
        self.offset_dec = nn.Sequential(
            # SubpelConv2x(g_ch_y, g_ch_y, 3, padding=1),
            ResidualBlockUpsample(g_ch_y, g_ch_y),
            ResidualBlockUpsample(g_ch_y, g_ch_y),
            DepthConvBlock(g_ch_y, g_ch_y),
            nn.Conv2d(g_ch_y, 32, 3, 1, 1)
            )
        self.recon_frame= ReconGeneration()
        self.align = Local_Align()
        self.feature_extractor = FeatureExtractor(g_ch_d, g_ch_d)
        self.hyper_enc = nn.Sequential(
                            DepthConvBlock(g_ch_y, g_ch_z),
                            ResidualBlockWithStride2(g_ch_z, g_ch_z),
                            ResidualBlockWithStride2(g_ch_z, g_ch_z),
                            DepthConvBlock(g_ch_y, g_ch_z),
                            DepthConvBlock(g_ch_y, g_ch_z),
                        )
        self.hyper_dec = nn.Sequential(
                            ResidualBlockUpsample(g_ch_z, g_ch_z),
                            ResidualBlockUpsample(g_ch_z, g_ch_z),
                            DepthConvBlock(g_ch_z, g_ch_y),
                            )
        self.temporal_enc = nn.Sequential(
            DepthConvBlock(g_ch_y * 2, g_ch_y, inplace=inplace),
            # DepthConvBlock(g_ch_y, g_ch_y, inplace=inplace),
        )
        self.prior_fusion_1 = nn.Sequential(
            DepthConvBlock(g_ch_y * 2, g_ch_y*2, inplace=inplace),
            # DepthConvBlock(g_ch_y*3, g_ch_y*2, inplace=inplace),
        )

        self.enc_Vector = nn.Parameter(torch.ones((1, g_ch_d, 1, 1)))
        self.num_enc_V = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))

        self.dec_Vector = nn.Parameter(torch.ones((1, g_ch_d, 1, 1)))
        self.num_dec_V = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        
        self.rec_Vector = nn.Parameter(torch.ones((1, g_ch_recon, 1, 1)))
        self.num_rec_V = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        
        self.tem_Vector = nn.Parameter(torch.ones((1, g_ch_d, 1, 1)))
        self.num_tem_V = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.hyper_Vector1 = nn.Parameter(torch.ones((1, g_ch_z, 1, 1)))
        self.num_hyper_V1 = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))

        self.hyper_Vector2 = nn.Parameter(torch.ones((1, g_ch_z, 1, 1)))
        self.num_hyper_V2 = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))

        self.hyper_offset1 = nn.Parameter(torch.ones((1, g_ch_z, 1, 1)))
        self.num_offset1 = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.hyper_offset2 = nn.Parameter(torch.ones((1, g_ch_z, 1, 1)))
        self.num_offset2 = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))

        self.lpips_fn = lpips.LPIPS(net='alex')

    def get_q_for_inference(self,  q_index):
        enc_Vector = self.num_enc_V
        vector_enc = self.get_curr_q(enc_Vector, self.enc_Vector, q_index=q_index)
        
        dec_Vector = self.num_dec_V
        vector_dec = self.get_curr_q(dec_Vector, self.dec_Vector, q_index=q_index)
        
        rec_Vector = self.num_rec_V
        vector_rec = self.get_curr_q(rec_Vector, self.rec_Vector, q_index=q_index)
        
        tem_Vector = self.num_tem_V
        vector_tem = self.get_curr_q(tem_Vector, self.tem_Vector, q_index=q_index)
        
        hyper_Vector1 = self.num_hyper_V1
        vector_hyper1 = self.get_curr_q(hyper_Vector1, self.hyper_Vector1, q_index=q_index)
        
        hyper_Vector2 = self.num_hyper_V2
        vector_hyper2 = self.get_curr_q(hyper_Vector2, self.hyper_Vector2, q_index=q_index)

        offset_Vector1 = self.num_offset1
        vector_offset1 = self.get_curr_q(offset_Vector1, self.hyper_offset1, q_index=q_index)
        
        offset_Vector2 = self.num_offset2
        vector_offset2 = self.get_curr_q(offset_Vector2, self.hyper_offset2, q_index=q_index)

        return vector_enc,vector_dec,vector_rec,vector_tem,vector_hyper1,vector_hyper2,vector_offset1,vector_offset2
    
    def get_offset_enc_dec(self, ref_feature,feature_8,vector_offset1,vector_offset2):
        d_ref_feature = F.interpolate(ref_feature,scale_factor=0.5,mode='bilinear')
        d_feature = F.interpolate(feature_8,scale_factor=0.5,mode='bilinear')
        offset = self.Offsetnet(torch.cat([d_ref_feature,d_feature],dim=1))
        offset_latent = self.offset_enc(offset)
        offset_latent_scale = offset_latent*vector_offset1
        offset_latent_q = self.quant(offset_latent_scale)
        offset_latent_hat = offset_latent_q*vector_offset2
        offset_hat = 6.0 * torch.tanh(
                            self.offset_dec(offset_latent_hat)
                        )
        offset_hat = F.interpolate(offset_hat,scale_factor=2,mode='bilinear')
        return offset_hat,offset_latent_q

    def forward_one_frame(self, x, dpb, frame_idx,use_lpip):
        _, _, H, W = x.size()
        pixel_num = H * W
        vector_enc,vector_dec,vector_rec,vector_tem,vector_hyper1, vector_hyper2,vector_offset1,vector_offset2 = self.get_q_for_inference(frame_idx)
        
        feature = F.pixel_unshuffle(x, 8)
        feature_8 = self.Feature_adaptor_X(x)
        feature_I = F.pixel_unshuffle(dpb["ref_frame"], 8)
        if dpb['ref_feature'] is not None:
            ref_feature = dpb['ref_feature']
        else:
            ref_feature = self.Feature_adaptor_I(feature_I) #1/8

        offset_hat,offset_latent_q = self.get_offset_enc_dec(ref_feature,feature_8,vector_offset1,vector_offset2)
        ctx = self.align(ref_feature,offset_hat)

        F_tem,F_tem_e = self.feature_extractor(ctx,vector_tem,frame_idx)
        y = self.encoder(feature,feature_8,F_tem,vector_enc)
        z= self.hyper_enc(y)
        z_scale = z*vector_hyper1
        z_q= self.quant(z_scale )
        z_scale_hat = z_q* vector_hyper2
        hyper_param = self.hyper_dec(z_scale_hat)
        if dpb["ref_y"] is not None:
            ref_y = dpb["ref_y"]
            tem_param = torch.cat([F_tem_e,ref_y],dim=1)

        else:
            indentity = torch.zeros_like(F_tem_e)
            tem_param = torch.cat([F_tem_e,indentity],dim=1)
        tem_param = self.temporal_enc(tem_param)
        param = self.prior_fusion_1(torch.cat([hyper_param,tem_param],dim =1))
        scale,mean = param.chunk(2,1)
        scale = self.scale_bound(scale.abs())

        y_res = y-mean
        y_q = self.quant(y_res)
        y_hat = y_q + mean

        F_t = self.decoder(y_hat,F_tem,vector_dec)
        
        x_rec = self.recon_frame(F_t,vector_rec)
        y_for_bit = y_q
        z_for_bit = z_q
        offset_for_bit = offset_latent_q
        bits_y = self.get_y_laplace_bits(y_for_bit, scale)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_offset = self.get_z_bits(offset_for_bit, self.bit_estimator_offset)
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_offset = torch.sum(bits_offset, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_z+bpp_offset
        mse_loss = torch.mean((x_rec - x).pow(2))
        yuv_loss = mse_loss
        lpips_loss = yuv_loss
        ssim_loss = yuv_loss        
        if use_lpip:
            lpips_loss = self.lpips_fn(x_rec, x, normalize=True).mean()
            ssim_loss = 1-ms_ssim(x_rec, x,data_range=1.0).mean()
            
        return {
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                'bpp_offset':bpp_offset,
                "bpp": bpp,
                "mse_loss": mse_loss,
                "yuv_loss":{"y_loss":yuv_loss,
                            "u_loss":yuv_loss,
                             "v_loss":yuv_loss,
                              "yuv_loss":yuv_loss,
                              },
                "lpips_loss":lpips_loss,
                "ssim_loss": ssim_loss,
                "dpb": {
                    "ref_frame": x_rec,
                    "ref_feature": F_t,
                    "ref_feature_L":F_t,
                    "ref_y": y_hat,

                },
                "bit_y": bits_y,
                "bit_z": bits_z,
                }
        
    def cal_macs(self):
        from ..utils.common import flops_calculator
        vector_enc,vector_dec,vector_rec,vector_tem,vector_hyper1, vector_hyper2,vector_offset1,vector_offset2 =  self.get_q_for_inference(0)
        x = torch.randn(1, 3, 256, 256)
        f = torch.randn(1, g_ch_src_d, 32, 32)
        f1 = torch.randn(1, g_ch_d, 32, 32)

        offset = torch.randn(1,32, 32, 32)
        offset_latent = torch.randn(1,g_ch_y, 4, 4)
        y = torch.randn(1, g_ch_y, 16, 16)
        z = torch.randn(1, g_ch_z, 4, 4)
        f4 = torch.randn(1, g_ch_y*2, 16, 16)
        f5 = torch.randn(1, g_ch_y*2, 32, 32)
        pixels = 256 * 256
        
        
        msgs = {

            'Feature_adaptor_X': flops_calculator(pixels,  self.Feature_adaptor_X, x),
            'Feature_adaptor_I': flops_calculator(pixels,  self.Feature_adaptor_I, f),
            'feature_extractor': flops_calculator(pixels,  self.feature_extractor, f1,vector_tem),
            'encoder': flops_calculator(pixels,  self.encoder, f,f1,f1,vector_enc),
            'hyper_enc': flops_calculator(pixels,  self.hyper_enc, y),
            'hyper_dec': flops_calculator(pixels,  self.hyper_dec, z),
            'offset_net': flops_calculator(pixels,  self.Offsetnet,f4),
            'offset_enc': flops_calculator(pixels,  self.offset_enc,y),
            'offset_dec': flops_calculator(pixels,  self.offset_dec,offset_latent),
            'align': flops_calculator(pixels,  self.align,f1,offset),
            'prior_fusion_1': flops_calculator(pixels,  self.prior_fusion_1, f4),
            'temporal_enc': flops_calculator(pixels,  self.temporal_enc, f4),
            'decoder': flops_calculator(pixels,  self.decoder, y,f1,vector_dec),
            'recon_frame': flops_calculator(pixels,  self.recon_frame, f1,vector_rec),  
        }
        enc_models_i = ['Feature_adaptor_X','Feature_adaptor_I', 'feature_extractor', 'encoder','decoder','hyper_enc','hyper_dec','temporal_enc','prior_fusion_1', 'offset_enc', 'offset_dec','offset_net','align']
        enc_models =  ['Feature_adaptor_X', 'feature_extractor', 'encoder','decoder','hyper_enc','hyper_dec','temporal_enc','prior_fusion_1', 'offset_enc', 'offset_dec','offset_net','align']
        dec_models_i = ['Feature_adaptor_I', 'feature_extractor','decoder','prior_fusion_1','hyper_dec','recon_frame','temporal_enc', 'offset_dec','offset_net','align']
        dec_models_p = ['feature_extractor','decoder','prior_fusion_1','hyper_dec','recon_frame','temporal_enc', 'offset_dec','offset_net','align']
        total_Enc_flops = 0
        total_Enc_i_flops = 0
        total_Dec_i_flops = 0
        total_Dec_p_flops = 0
        for key, flop in msgs.items():
            if key in enc_models:
                if 'K' in flop:
                    val = float(flop[:-1])
                    val = val * 1000
                else:
                    val = float(flop[:-5])
                total_Enc_flops += val
            if key in enc_models_i:
                if 'K' in flop:
                    val = float(flop[:-1])
                    val = val * 1000
                else:
                    val = float(flop[:-5])
                total_Enc_i_flops += val
            if key in dec_models_i:
                if 'K' in flop:
                    val = float(flop[:-1])
                    val = val * 1000
                else:
                    val = float(flop[:-5])
                total_Dec_i_flops += val
            if key in dec_models_p:
                if 'K' in flop:
                    val = float(flop[:-1])
                    val = val * 1000
                else:
                    val = float(flop[:-5])
                total_Dec_p_flops += val
        msgs['Enc'] = str(total_Enc_flops / 1000) + ' KMac'
        msgs['Enc_i'] = str(total_Enc_i_flops / 1000) + ' KMac'
        msgs['Dec_i'] = str(total_Dec_i_flops / 1000) + ' KMac'
        msgs['Dec_p'] = str(total_Dec_p_flops / 1000) + ' KMac'
        return msgs
    
class Multi_DMC(nn.Module):
    def __init__(self,  inplace=False):
        super().__init__()
        self.net0 = DMC()
        self.net1 = DMC()
        self.net2 = DMC()
    def load_pretrained_weights(self, pretrained_path):
        """从预训练DMC模型加载权重到所有子网络"""
        # 加载预训练权重
        load_model_para = torch.load(pretrained_path, map_location='cpu')['state_dict']
        
        # 为每个子网络加载相同的权重
        for i in range(3):
            net = getattr(self, f'net{i}')
            model_dict = net.state_dict()
            load_model_para = {k: v for k, v in load_model_para.items() if k in model_dict}
            model_dict.update(load_model_para )
            net.load_state_dict(model_dict, strict = False)
            print(f"Loaded pretrained weights to net{i}")   
             
    def forward_one_frame(self, x, dpb, frame_idx=0,use_lip=False):
        if frame_idx ==0:
            result = self.net0.forward_one_frame(x, dpb,frame_idx=0, use_lpip = use_lip)
        elif frame_idx ==2:
            result = self.net1.forward_one_frame(x, dpb,frame_idx=2, use_lpip = use_lip)
        else:
            result = self.net2.forward_one_frame(x, dpb,frame_idx=1, use_lpip = use_lip)
        return result
    def cal_macs(self):
        result = self.net0.cal_macs()
        return result