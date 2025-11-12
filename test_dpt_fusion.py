#!/usr/bin/env python3
"""
测试DPT融合功能的简单脚本
"""

import torch
import torch.nn as nn
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.enhanced_losses import EnhancedDPTLoss
from omegaconf import DictConfig

def test_dpt_fusion():
    """测试DPT融合功能"""
    
    # 创建测试配置
    cfg = DictConfig({
        'model': {
            'name': 'SingleUNet',  # 添加缺失的name字段
            'use_dpt_fusion': True,
            'dpt_model_path': "/202421000505/wsh_project/refine_splatter/splatter-image_dpt/scene/model_cache/Depth-Anything-v2",
            'network_with_offset': True,
            'network_without_offset': False,
            'base_dim': 64,
            'num_blocks': 2,
            'attention_resolutions': [16],
            'max_sh_degree': 0,
            'isotropic': False,
            'cross_view_attention': False,
            'inverted_x': False,
            'inverted_y': False,
            'depth_scale': 1.0,
            'xyz_scale': 1.0,
            'opacity_scale': 1.0,
            'scale_scale': 1.0,
            'depth_bias': 0.0,
            'xyz_bias': 0.0,
            'opacity_bias': 0.0,
            'scale_bias': 1.0,
        },
        'data': {
            'training_resolution': 64,
            'category': 'objaverse',
            'white_background': True,
            'origin_distances': False,
            'fov': 49.134342641202636,
            'znear': 0.8,
            'zfar': 3.2,
        },
        'cam_embd': {
            'embedding': None,
        }
    })
    
    # 测试DPT特征提取器
    print("测试DPT特征提取器...")
    from scene.feature_fusion import DPTFeatureExtractor
    
    dpt_extractor = DPTFeatureExtractor()
    print(f"DPT模型是否加载成功: {dpt_extractor.dpt_model is not None}")
    
    if dpt_extractor.dpt_model is not None:
        print("✅ 成功加载真正的DPT模型！")
        print(f"DPT模型类型: {type(dpt_extractor.dpt_model)}")
    else:
        print("⚠️  使用自定义特征提取器作为备选")
    
    # 测试特征提取
    test_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        features = dpt_extractor(test_input)
        print(f"DPT特征形状: {features.shape}")
    
    # 创建模型
    print("\n创建GaussianSplatPredictor...")
    model = GaussianSplatPredictor(cfg)
    model.eval()
    
    # 创建DPT损失函数
    print("创建DPT损失函数...")
    dpt_loss_fn = EnhancedDPTLoss(
        depth_weight=0.1,
        consistency_weight=0.05,
        fusion_weight=0.1
    )
    
    # 创建测试数据
    batch_size = 2
    num_views = 3
    height, width = 64, 64
    
    print("创建测试数据...")
    input_images = torch.randn(batch_size, num_views, 3, height, width)
    view_to_world_transforms = torch.randn(batch_size, num_views, 4, 4)
    source_cv2wT_quat = torch.randn(batch_size, num_views, 4)
    
    # 测试不使用DPT特征的情况
    print("测试不使用DPT特征...")
    with torch.no_grad():
        gaussian_splats = model(input_images, view_to_world_transforms, source_cv2wT_quat)
        print(f"输出形状: {[(k, v.shape) for k, v in gaussian_splats.items()]}")
    
    # 测试使用DPT特征的情况
    print("测试使用DPT特征...")
    with torch.no_grad():
        gaussian_splats, dpt_features_dict = model(
            input_images, 
            view_to_world_transforms, 
            source_cv2wT_quat,
            return_dpt_features=True
        )
        print(f"输出形状: {[(k, v.shape) for k, v in gaussian_splats.items()]}")
        print(f"DPT特征形状: {[(k, v.shape) for k, v in dpt_features_dict.items()]}")
    
    # 测试DPT损失计算
    print("测试DPT损失计算...")
    dpt_features = dpt_features_dict['dpt_features']
    unet_features = dpt_features_dict['unet_features']
    fused_features = dpt_features_dict['fused_features']
    
    dpt_loss, loss_dict = dpt_loss_fn(
        pred_depth=None,
        gt_depth=None,
        unet_features=unet_features,
        dpt_features=dpt_features,
        fused_features=fused_features
    )
    
    print(f"DPT损失: {dpt_loss.item():.6f}")
    print(f"损失详情: {loss_dict}")
    
    print("\n✅ 所有测试通过！DPT融合功能正常工作。")
    if dpt_extractor.dpt_model is not None:
        print("✅ 使用的是真正的DPT模型进行特征提取！")
    else:
        print("⚠️  使用的是自定义特征提取器，建议检查DPT模型路径")

if __name__ == "__main__":
    test_dpt_fusion()