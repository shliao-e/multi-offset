import torch
import torch.nn.functional as F
from torch import nn
from ..transforms.functional import ycbcr2rgb
from .common_model import CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2

# qp_shift = [0, 8, 4]
# extra_qp = max(qp_shift)

g_ch_src_d = 3 * 8 * 8
g_ch_recon = 320
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, feature, ctx, quant_step):
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature


class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefFrame():
    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='laplace', z_channel=g_ch_z, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)
        # self.qp_shift = qp_shift

        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor()

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()

        self.q_encoder = nn.Parameter(torch.ones((4, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((4, g_ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((4, g_ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((4, g_ch_recon, 1, 1)))



    def apply_feature_adaptor(self,dpb):
        if dpb['ref_feature'] is None:
            return self.feature_adaptor_i(F.pixel_unshuffle(dpb['ref_frame'], 8))
        return self.feature_adaptor_p( dpb['ref_feature'])

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def prepare_feature_adaptor_i(self, last_qp):
        if self.dpb[0].frame is None:
            q_recon = self.q_recon[last_qp:last_qp+1, :, :, :]
            self.dpb[0].frame = self.recon_generation_net(self.dpb[0].feature, q_recon).clamp_(0, 1)
            self.reset_ref_feature()
    def forward(self, x, dpb, qp,use_lpip,use_yuv):
        device = x.device
        _, _, H, W = x.size()
        pixel_num = H * W
        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon = self.q_recon[qp:qp+1, :, :, :]
        x_d = F.pixel_unshuffle(x, 8)
        feature = self.apply_feature_adaptor(dpb)
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x_d, ctx, q_encoder)
        hyper_inp,_ = self.pad_for_y(y)
        z = self.hyper_encoder(hyper_inp)
        z_hat= self.quant(z)
        param = self.res_prior_param_decoder(z_hat, ctx_t)
        quant,scale,mean = param.chunk(3,1)
        quant = torch.clamp_min(quant, 0.5)
        y = y / quant
        y_res = y-mean
        y_q = self.quant(y_res)
        y_hat = y_q + mean
        y_hat = y_hat * quant 
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
         
        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scale)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_z
        y_loss = torch.mean((x_hat[:,0,:,:] - x[:,0,:,:]).pow(2))
        u_loss = torch.mean((x_hat[:,1,:,:] - x[:,1,:,:]).pow(2))
        v_loss = torch.mean((x_hat[:,2,:,:] - x[:,2,:,:]).pow(2))
        yuv_loss = (6*y_loss+u_loss+v_loss)/8
        mse_loss = yuv_loss
        lpips_loss = yuv_loss
        if use_lpip:
            rgb  = ycbcr2rgb(x)
            rec_rgb = ycbcr2rgb(x_hat)
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
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_y": y_hat,

                },
                "bit_y": bits_y,
                "bit_z": bits_z,
                }
    

    def cal_macs(self):
        from ..utils.common import flops_calculator
        q_encoder = self.q_encoder[0, :, :, :]
        q_decoder = self.q_decoder[0, :, :, :]
        q_feature = self.q_feature[0, :, :, :]
        q_recon = self.q_recon[0, :, :, :]
        x = torch.randn(1, 8*8*3,32, 32)
        f1 = torch.randn(1, g_ch_d, 32, 32)
        y = torch.randn(1, g_ch_y, 16, 16)
        z = torch.randn(1, g_ch_z, 4, 4)
        f4 = torch.randn(1, g_ch_y*3,16, 16)
        pixels = 256 * 256
        
        
        msgs = {
            'feature_adaptor_i': flops_calculator(pixels,  self.feature_adaptor_i, x),
            'feature_adaptor_p': flops_calculator(pixels,  self.feature_adaptor_p, f1),
            'feature_extractor': flops_calculator(pixels,  self.feature_extractor, f1,q_feature),
            'encoder': flops_calculator(pixels,  self.encoder, x,f1,q_encoder),
            'hyper_enc': flops_calculator(pixels,  self.hyper_encoder, y),
            'hyper_dec': flops_calculator(pixels,  self.hyper_decoder, z),
            'temporal_prior_encoder': flops_calculator(pixels,  self.temporal_prior_encoder, f1),
            'y_prior_fusion': flops_calculator(pixels,  self.y_prior_fusion, f4),
            'decoder': flops_calculator(pixels,  self.decoder, y,f1,q_decoder),
            'recon_generation_net': flops_calculator(pixels,  self.recon_generation_net, f1,q_recon),  
        }
        enc_models_i = ['feature_adaptor_i','y_prior_fusion', 'feature_extractor','encoder','hyper_enc']
        enc_models =  ['feature_adaptor_p','y_prior_fusion', 'feature_extractor','encoder','hyper_enc']
        dec_models_i = ['feature_adaptor_i','y_prior_fusion', 'feature_extractor','decoder','hyper_dec','recon_generation_net']
        dec_models_p = ['feature_adaptor_p','y_prior_fusion', 'feature_extractor','decoder','hyper_dec','recon_generation_net']
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

