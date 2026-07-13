import os
import torch

import imageio
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from ..utils.info import classes_dict
from src.transforms.functional import rgb_to_ycbcr,ycbcr420_to_rgb

class CTS(data.Dataset):
    def __init__(self, root_dir, test_class,output_yuv = False):
        self.test_class = test_class
        self.clip = []
        self.output_yuv = output_yuv
        for i, seq in enumerate(classes_dict[test_class]["sequence_name"]):
            v_frames = classes_dict[test_class]["frameNum"][i]
            i_frame_path = []
            frame_path = []
            for j in range(v_frames):
                frame_path.append(os.path.join(root_dir, test_class,"images_crop/" + seq, str(j + 1).zfill(3) + '.png'))
            self.clip.append((i_frame_path, frame_path))
    def __len__(self):
        return len(self.clip)
    
    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        if self.output_yuv:
           img =rgb_to_ycbcr(img)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img[:, :, : ]

    def __getitem__(self, index):
        index = index % len(self.clip)
        frames = [self.read_img(img_path) for img_path in self.clip[index][1]]
        frames = torch.stack(frames, 0)
        return {
            "yuv444":frames ,
            'frames': frames
        } 
    
class CTS_yuv(data.Dataset):
    def __init__(self, root_dir, test_class,output_rgb = False):
        self.test_class = test_class
        self.clip = []
        self.output_rgb = output_rgb
        # 构建clip列表：每个clip包含(yuv文件路径, 帧数)
        for i, yuv_filename in enumerate(classes_dict[test_class]["ori_yuv"]):
            yuv_path = os.path.join(root_dir, 'HEVC',test_class,'org' ,yuv_filename)
            yuv_resolution = classes_dict[test_class]["yuv_resolution"]
            if 'AVS' in test_class:
                yuv_path = os.path.join(root_dir,'AVS3_Seq',yuv_filename)
            # 解析分辨率
                yuv_resolution = classes_dict[test_class]["yuv_resolution"]
            elif "huawei"in test_class:
                yuv_path = os.path.join(root_dir,'huawei-test2',yuv_filename)
                yuv_resolution = classes_dict[test_class]["yuv_resolution"][i]
            elif "DJL"in test_class:
                yuv_path = os.path.join(root_dir,test_class,"960x544p10fps",yuv_filename)
            # 解析分辨率
                yuv_resolution = classes_dict[test_class]["yuv_resolution"][i]
            width, height = map(int, yuv_resolution.split('x'))
            frame_num = classes_dict[test_class]["frameNum"][i]
            self.clip.append((yuv_path, frame_num,width,height))
    
    def __len__(self):
        return len(self.clip)
    
    def read_yuv_frames(self, yuv_path, frame_num,width,height):
        """
        一次性读取所有帧的YUV数据（效率更高）
        
        Returns:
            y_frames: [T, H, W] uint8 Y分量
            u_frames: [T, H//2, W//2] uint8 U分量  
            v_frames: [T, H//2, W//2] uint8 V分量
        """
        # 计算YUV420帧大小
        y_size = width * height
        uv_size = y_size // 4
        frame_size = y_size + 2 * uv_size
        # 预分配内存
        y_frames = np.zeros((frame_num, height, width), dtype=np.uint8)
        u_frames = np.zeros((frame_num, height//2, width//2), dtype=np.uint8)
        v_frames = np.zeros((frame_num, height//2, width//2), dtype=np.uint8)
        
        with open(yuv_path, 'rb') as f:
            for i in range(frame_num):
                f.seek(i * frame_size)
                
                # 读取Y分量
                y_data = np.frombuffer(f.read(y_size), dtype=np.uint8)
                y_frames[i] = y_data.reshape(height, width)
                
                # 读取U分量
                u_data = np.frombuffer(f.read(uv_size), dtype=np.uint8)
                u_frames[i] = u_data.reshape(height//2, width//2)
                
                # 读取V分量
                v_data = np.frombuffer(f.read(uv_size), dtype=np.uint8)
                v_frames[i] = v_data.reshape(height//2, width//2)
        
        return y_frames, u_frames, v_frames
    
    def __getitem__(self, index):
        yuv_path, frame_num,width,height = self.clip[index]
        frames =[]
        # 读取YUV数据
        y_frames, u_frames, v_frames = self.read_yuv_frames(yuv_path, frame_num,width,height)
        
        # 转换为tensor（保持uint8，后续转换时再归一化）
        y_tensor = torch.from_numpy(y_frames).unsqueeze(1)  # [T, 1, H, W]
        u_tensor = torch.from_numpy(u_frames).unsqueeze(1)  # [T, 1, H//2, W//2]
        v_tensor = torch.from_numpy(v_frames).unsqueeze(1)  # [T, 1, H//2, W//2]
        u_upsampled = F.interpolate(u_tensor, size=y_tensor.shape[-2:], mode='bilinear', align_corners=False)
        v_upsampled = F.interpolate(v_tensor, size=y_tensor.shape[-2:], mode='bilinear', align_corners=False)
        # 合并为YUV444
        yuv_444 = torch.cat([y_tensor, u_upsampled, v_upsampled], dim=1)  # 在通道维度合并
        if self.output_rgb:
            frames = ycbcr420_to_rgb(y_tensor/255,u_tensor/255,v_tensor/255)
        # 可选：归一化到[0, 1]
        yuv_444 = yuv_444.float() / 255.0
        
        # 返回YUV分量 + 分辨率信息（用于后续上采样）
        return {
            'frames': frames,
            'y': y_tensor,
            'u': u_tensor,
            'v': v_tensor,
            "yuv444":yuv_444,
            'file_path': yuv_path
        }   
    

class CTS_con(data.Dataset):
    def __init__(self, root_dir, test_class, sec_id = 0,output_yuv = False):
        self.test_class = test_class
        self.clip = []
        self.sec_id = sec_id
        self.output_yuv = output_yuv
        seq =classes_dict[test_class]["sequence_name"][self.sec_id]
        v_frames = classes_dict[test_class]["frameNum"][sec_id]
        frame_path = []
        for j in range(v_frames):
            frame_path.append(os.path.join(root_dir,test_class+"/images_crop",seq, str(j + 1).zfill(3) + '.png'))
        self.clip.append(frame_path)
    
    def __len__(self):
        return len(self.clip)
    
    def get_intra_bits(self, bin_path):
        bits = os.path.getsize(bin_path) * 8
        return bits

    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        if self.output_yuv:
           img =rgb_to_ycbcr(img)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img[:, :, : ]

    def __getitem__(self, index):
        index = index % len(self.clip)
        frames = [self.read_img(img_path) for img_path in self.clip[index]]
        frames = torch.stack(frames, 0)
        return {
            "yuv444":frames ,
            'frames': frames
        } 