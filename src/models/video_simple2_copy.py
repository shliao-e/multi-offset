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
from ..transforms.functional import ycbcr2rgb,yuv_444_to_420
import lpips

g_ch_src_d = 3 * 8 * 8
g_ch_recon = 192
g_ch_y = 128
g_ch_z = 128
g_ch_d = 128


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
        self.adaptor = nn.Conv2d(channel,g_ch_y,3,2,1)
        # self.adaptor =nn.ModuleList([
        #                             nn.Conv2d(channel,g_ch_y,3,2,1)
        #                                  for _ in range (4)])  
    def forward(self, feature,quant_step=None,frame_idx=0):
        layer1 = self.layer1(feature)
        out1 = layer1 * quant_step
        out1= self.adaptor(out1)
        out = self.layer2(layer1)
        return out, out1
class Multi_scale_ref_fusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        # self.conv1_down = nn.Conv2d(g_ch_d//2,g_ch_d,3, stride=2, padding=1)
        self.block1 = DepthConvBlock(g_ch_d//2,g_ch_d,stride=2,inplace=inplace)
        
        self.conv2_out = nn.Conv2d(g_ch_d*2,g_ch_d,3, stride=1, padding=1)
        self.block2 = DepthConvBlock(g_ch_d,g_ch_d, inplace=inplace)
        

    def forward(self, feature_1_4, feature_1_8):
        # feature_1_4_out = self.conv1_down(feature_1_4)
        feature_1_4_out = self.block1(feature_1_4)
        
        feature_1_8_out = self.conv2_out(torch.cat([feature_1_4_out,feature_1_8],dim = 1))
        feature_1_8_out = self.block2(feature_1_8_out)
        feature_1_8 = feature_1_8+feature_1_8_out
        return feature_1_8
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
        self.recon_f = nn.Sequential(
            ResidualBlockUpsample(g_ch_d,g_ch_d//2 , 2),)
            # DepthConvBlock(g_ch_d//2,g_ch_d//2))
    def forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        feature_up  = self.recon_f(feature)
        return feature,feature_up
    
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
        self.Multi_fusion = Multi_scale_ref_fusion()
        self.Feature_adaptor_I = nn.Sequential(
            DepthConvBlock(g_ch_src_d, g_ch_d, inplace=inplace),
            DepthConvBlock(g_ch_d, g_ch_d, inplace=inplace),
        )

        self.encoder = encoder(g_ch_d, g_ch_y, inplace=inplace)
        self.decoder = decoder()
        
        self.recon_frame= ReconGeneration()
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
        
        
        return vector_enc,vector_dec,vector_rec,vector_tem
        
    def forward_one_frame(self, x, dpb, frame_idx,use_lpip,use_yuv):
        _, _, H, W = x.size()
        pixel_num = H * W
        vector_enc,vector_dec,vector_rec,vector_tem= self.get_q_for_inference(frame_idx)
        
        feature = F.pixel_unshuffle(x, 8)
        feature_8 = self.Feature_adaptor_X(x)
        if dpb['ref_feature'] is not None:
            ref_feature = self.Multi_fusion(dpb['ref_feature_L'],dpb['ref_feature'])
        else:
            ref_feature = F.pixel_unshuffle(dpb["ref_frame"], 8)
            ref_feature = self.Feature_adaptor_I(ref_feature) #1/8
        F_tem,F_tem_e = self.feature_extractor(ref_feature,vector_tem,frame_idx)
        y = self.encoder(feature,feature_8,F_tem,vector_enc)
        z= self.hyper_enc(y)

        z_q= self.quant(z )

        hyper_param = self.hyper_dec(z_q)
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
        # quant = torch.clamp_min(quant, 0.5)
        # quant = self.quant_bound(quant)
        # y = y / quant
        y_res = y-mean
        y_q = self.quant(y_res)
        y_hat = y_q + mean
        # y_hat = y_hat * quant    
        F_t, F_t_1_4 = self.decoder(y_hat,F_tem,vector_dec)
        
        x_rec = self.recon_frame(F_t,vector_rec)
        y_for_bit = y_q
        z_for_bit = z_q
        bits_y = self.get_y_laplace_bits(y_for_bit, scale)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_z
        recon_y_norm, recon_u_norm, recon_v_norm = yuv_444_to_420(x_rec)
        ori_y_norm,ori_u_norm, ori_v_norm = yuv_444_to_420(x)

        y_loss = torch.mean((recon_y_norm -ori_y_norm).pow(2))
        u_loss = torch.mean((recon_u_norm -ori_u_norm).pow(2))
        v_loss = torch.mean((recon_v_norm -ori_v_norm).pow(2))
        yuv_loss = (4*y_loss+u_loss+v_loss)/6
        mse_loss = yuv_loss
        lpips_loss = yuv_loss
        if use_lpip:
            rgb  = ycbcr2rgb(x)
            rec_rgb = ycbcr2rgb(x_rec)
            lpips_loss = self.lpips_fn(rec_rgb, rgb).mean()
            
        return {
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "mse_loss": mse_loss,
                "yuv_loss":{"y_loss":y_loss,
                            "u_loss":u_loss,
                             "v_loss":v_loss,
                              "yuv_loss":yuv_loss,
                              },
                "lpips_loss":lpips_loss,
                "dpb": {
                    "ref_frame": x_rec,
                    "ref_feature": F_t,
                    "ref_feature_L":F_t_1_4,
                    "ref_y": y_hat,

                },
                "bit_y": bits_y,
                "bit_z": bits_z,
                }
        
    def cal_macs(self):
        from ..utils.common import flops_calculator
        vector_enc,vector_dec,vector_rec,vector_tem=  self.get_q_for_inference(0)
        x = torch.randn(1, 3, 256, 256)
        f = torch.randn(1, g_ch_src_d, 32, 32)
        f1 = torch.randn(1, g_ch_d, 32, 32)
        f3 = torch.randn(1,g_ch_d//2, 64, 64)
        y = torch.randn(1, g_ch_y, 16, 16)
        z = torch.randn(1, g_ch_z, 4, 4)
        f4 = torch.randn(1, g_ch_y*2, 16, 16)
        f5 = torch.randn(1, g_ch_y*3, 16, 16)
        pixels = 256 * 256
        
        
        msgs = {
            'Multi_fusion':flops_calculator(pixels,  self.Multi_fusion, f3,f1),
            'Feature_adaptor_X': flops_calculator(pixels,  self.Feature_adaptor_X, x),
            'Feature_adaptor_I': flops_calculator(pixels,  self.Feature_adaptor_I, f),
            'feature_extractor': flops_calculator(pixels,  self.feature_extractor, f1,vector_tem),
            'encoder': flops_calculator(pixels,  self.encoder, f,f1,f1,vector_enc),
            'hyper_enc': flops_calculator(pixels,  self.hyper_enc, y),
            'hyper_dec': flops_calculator(pixels,  self.hyper_dec, z),
            'prior_fusion_1': flops_calculator(pixels,  self.prior_fusion_1, f4),
            'temporal_enc': flops_calculator(pixels,  self.temporal_enc, f4),
            'decoder': flops_calculator(pixels,  self.decoder, y,f1,vector_dec),
            'recon_frame': flops_calculator(pixels,  self.recon_frame, f1,vector_rec),  
        }
        enc_models_i = ['Feature_adaptor_X','Feature_adaptor_I', 'feature_extractor', 'encoder','decoder','hyper_enc','hyper_dec','temporal_enc','prior_fusion_1']
        enc_models =  ['Feature_adaptor_X','Multi_fusion', 'feature_extractor', 'encoder','decoder','hyper_enc','hyper_dec','temporal_enc','prior_fusion_1']
        dec_models_i = ['Feature_adaptor_I', 'feature_extractor','decoder','prior_fusion_1','hyper_dec','recon_frame','temporal_enc']
        dec_models_p = ['feature_extractor','decoder','prior_fusion_1','hyper_dec','recon_frame','temporal_enc','Multi_fusion']
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