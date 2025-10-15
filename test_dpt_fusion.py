#!/usr/bin/env python3
"""
测试DPT特征融合功能的脚本
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.feature_fusion import DPTFeatureExtractor, DPTFeatureProjector, DPTUNetFusion
from scene.gaussian_predictor import SongUNet

def test_dpt_feature_extractor():
    """测试DPT特征提取器"""
    print("测试DPT特征提取器...")
    
    # 创建测试输入
    batch_size = 2
    height, width = 128, 128
    test_images = torch.randn(batch_size, 3, height, width)
    
    try:
        # 创建DPT特征提取器
        dpt_extractor = DPTFeatureExtractor()
        
        # 提取特征
        with torch.no_grad():
            dpt_features = dpt_extractor(test_images)
        
        print(f"DPT特征形状: {dpt_features.shape}")
        print("✓ DPT特征提取器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ DPT特征提取器测试失败: {e}")
        return False

def test_dpt_feature_projector():
    """测试DPT特征投影器"""
    print("测试DPT特征投影器...")
    
    # 创建测试输入
    batch_size = 2
    height, width = 32, 32
    dpt_features = torch.randn(batch_size, 1024, height, width)
    
    try:
        # 创建DPT特征投影器
        projector = DPTFeatureProjector(dpt_feature_dim=1024, unet_feature_dim=512)
        
        # 投影特征
        projected_features = projector(dpt_features)
        
        print(f"投影后特征形状: {projected_features.shape}")
        print("✓ DPT特征投影器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ DPT特征投影器测试失败: {e}")
        return False

def test_dpt_unet_fusion():
    """测试DPT-UNet融合模块"""
    print("测试DPT-UNet融合模块...")
    
    # 创建测试输入
    batch_size = 2
    height, width = 128, 128
    unet_features = torch.randn(batch_size, 512, 32, 32)  # UNet encoder最后一层特征
    input_images = torch.randn(batch_size, 3, height, width)
    
    try:
        # 创建DPT-UNet融合模块
        fusion_module = DPTUNetFusion(unet_feature_dim=512)
        
        # 融合特征
        fused_features = fusion_module(unet_features, input_images)
        
        print(f"融合后特征形状: {fused_features.shape}")
        print("✓ DPT-UNet融合模块测试通过")
        return True
        
    except Exception as e:
        print(f"✗ DPT-UNet融合模块测试失败: {e}")
        return False

def test_song_unet_with_dpt():
    """测试带DPT融合的SongUNet"""
    print("测试带DPT融合的SongUNet...")
    
    # 创建测试输入
    batch_size = 2
    height, width = 128, 128
    test_images = torch.randn(batch_size, 3, height, width)
    
    try:
        # 创建带DPT融合的SongUNet
        unet = SongUNet(
            img_resolution=128,
            in_channels=3,
            out_channels=64,
            model_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_blocks=4,
            attn_resolutions=[16],
            use_dpt_fusion=True,
            dpt_model_path="/202421000505/wsh_project/refine_splatter/splatter-image/scene/model_cache/dpt-large"
        )
        
        # 前向传播
        with torch.no_grad():
            output = unet(test_images, input_images=test_images)
        
        print(f"UNet输出形状: {output.shape}")
        print("✓ 带DPT融合的SongUNet测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 带DPT融合的SongUNet测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试DPT特征融合功能...")
    print("=" * 50)
    
    tests = [
        test_dpt_feature_extractor,
        test_dpt_feature_projector,
        test_dpt_unet_fusion,
        test_song_unet_with_dpt
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！DPT特征融合功能正常工作。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()



