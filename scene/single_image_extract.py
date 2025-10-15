import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import warnings
import torch.nn.functional as F
from feature_fusion import FeatureAlignAndFuse
import json

# 忽略警告信息
warnings.filterwarnings("ignore")

# 创建模型缓存目录
CACHE_DIR = "./model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_dino_model():
    """加载DINOv2模型，支持本地缓存"""
    dino_cache_path = os.path.join(CACHE_DIR, "dinov2_vits14.pth")
    
    if os.path.exists(dino_cache_path):
        print("从本地缓存加载DINOv2模型...")
        # 直接从网络加载模型架构
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", source='github')
        # 加载本地缓存的模型权重
        state_dict = torch.load(dino_cache_path)
        dino_model.load_state_dict(state_dict)
    else:
        print("从网络下载DINOv2模型...")
        # 从网络加载模型
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", source='github')
        # 保存模型权重到本地缓存
        torch.save(dino_model.state_dict(), dino_cache_path)
        print(f"DINOv2模型权重已缓存到: {dino_cache_path}")
    
    dino_model.eval()
    return dino_model

def load_dpt_model():
    """加载DPT模型，支持本地缓存"""
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    
    dpt_model_cache_path = os.path.join(CACHE_DIR, "dpt-large")
    dpt_config_path = os.path.join(dpt_model_cache_path, "config.json")
    
    if os.path.exists(dpt_config_path):
        print("从本地缓存加载DPT模型...")
        # 从本地加载模型
        dpt_model = DPTForDepthEstimation.from_pretrained(dpt_model_cache_path)
        dpt_image_processor = DPTImageProcessor.from_pretrained(dpt_model_cache_path)
    else:
        print("从网络下载DPT模型...")
        # 从网络加载模型
        dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        dpt_image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        
        # 创建目录并保存到本地缓存
        os.makedirs(dpt_model_cache_path, exist_ok=True)
        dpt_model.save_pretrained(dpt_model_cache_path)
        dpt_image_processor.save_pretrained(dpt_model_cache_path)
        print(f"DPT模型已缓存到: {dpt_model_cache_path}")
    
    dpt_model.eval()
    return dpt_model, dpt_image_processor

# ---- 1. DINOv2 ----
dino_model = load_dino_model()

# ---- 2. DPT-Large (深度估计) ----
dpt_model, dpt_image_processor = load_dpt_model()

# ---- 3. 加载测试图像 ----
image_path = "/202421000505/wsh_project/refine_splatter/Objaverse/views_release/08b5cec2bcda49628ad8ae96749b751a/001.png"  # 换成你自己的图片路径
image = Image.open(image_path).convert("RGB")

# ---- 4. 预处理 ----
# DINOv2 推荐 224×224 输入
transform_dino = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# DPT-Large 用 HuggingFace 的预处理器
inputs_dpt = dpt_image_processor(images=image, return_tensors="pt")

# ---- 5. 前向提取特征 ----
with torch.no_grad():
    # DINOv2
    img_dino = transform_dino(image).unsqueeze(0)  # [1,3,224,224]
    dino_feats = dino_model.get_intermediate_layers(img_dino, n=1)[0]  # [1, 197, 384]
    # 取去掉 CLS token 的 patch 特征并 reshape
    dino_feats = dino_feats[:, 1:, :]  # [1,196,384]
    
    # 获取特征的维度信息
    batch_size = dino_feats.shape[0]
    num_patches = dino_feats.shape[1]
    feature_dim = dino_feats.shape[2]
    
    # 计算特征图的高度和宽度（假设是正方形）
    h = w = int(math.sqrt(num_patches))
    
    # 验证计算是否正确
    if h * w != num_patches:
        # 如果不是正方形，尝试找到最接近的因数组合
        for i in range(int(math.sqrt(num_patches)), 0, -1):
            if num_patches % i == 0:
                h, w = i, num_patches // i
                break
    
    print(f"特征信息: batch_size={batch_size}, num_patches={num_patches}, feature_dim={feature_dim}")
    print(f"计算的特征图尺寸: h={h}, w={w}")
    
    # 正确reshape特征图
    dino_feats_map = dino_feats.reshape(batch_size, h, w, feature_dim).permute(0, 3, 1, 2)  # [1,384,h,w]

    # DPT
    outputs_dpt = dpt_model(**inputs_dpt)
    dpt_depth = outputs_dpt.predicted_depth  # [1,H,W]


# print(f"DINOv2 features shape: {dino_feats_map.shape}")  # e.g. [1,384,h,w]
# print(f"DPT depth shape: {dpt_depth.shape}")             # e.g. [1,H,W]

# =============== 接 SongUNet 提取 decoder 中间特征 ===============
from gaussian_predictor import SongUNet

# 构造一个最小的 SongUNet 以便拿到中间特征；输入需要 [B,3,Hs,Ws]
# 这里用与 DPT 相同的空间分辨率作为演示，你可以改为训练时的 cfg 分辨率
Hs = 256
Ws = 256

song_unet = SongUNet(
    img_resolution=Hs,
    in_channels=3,
    out_channels=3,
    emb_dim_in=0,
    model_channels=128,
    channel_mult=[1,2,2,2],
    num_blocks=2,
    attn_resolutions=[16]
)

with torch.no_grad():
    dummy_rgb = T.Resize((Hs, Ws))(transform_dino.transforms[1](image)).unsqueeze(0)  # [1,3,Hs,Ws]
    _, unet_feats = song_unet(dummy_rgb, return_decoder_feats=True)

# 1) 准备三路特征（以 U-Net 解码器特征分辨率为对齐目标）
B = dino_feats_map.shape[0]

# 确保 DPT 为 [B,1,H,W]
dpt_depth_4d = dpt_depth if dpt_depth.dim() == 4 else dpt_depth.unsqueeze(1)

# 2) 构造融合模块
fuser = FeatureAlignAndFuse(
    unet_channels=unet_feats.shape[1],
    dino_in_channels=dino_feats_map.shape[1],
    dpt_in_channels=dpt_depth_4d.shape[1],
    dino_proj_channels=128,
    dpt_embed_channels=64,
    fused_out_channels=256,
    use_attention=True
)

# 3) 前向融合
fused_feats, dino_up, dpt_emb = fuser(unet_feats, dino_feats_map, dpt_depth_4d)

# print("对齐后的形状：")
# print(f"unet_feats: {unet_feats.shape}")
# print(f"dino_up: {dino_up.shape}")     # [B,128,H,W]
# print(f"dpt_emb: {dpt_emb.shape}")     # [B,64,H,W]
# print(f"F_fused: {fused_feats.shape}") # [B,256,H,W]