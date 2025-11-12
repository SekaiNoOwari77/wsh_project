#!/usr/bin/env python3
"""
使用 Depth-Anything-v2 模型生成深度图的脚本
"""

import torch
import numpy as np
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import argparse


def generate_depth_map(
    image_path: str,
    model_path: str,
    output_path: str = None,
    device: str = None
):
    """
    使用 Depth-Anything-v2 模型生成深度图
    
    Args:
        image_path: 输入图片路径
        model_path: Depth-Anything-v2 模型路径
        output_path: 输出深度图路径（可选，默认为输入图片同目录下的 _depth.png）
        device: 设备（'cuda' 或 'cpu'，默认自动选择）
    """
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载图片
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"Original image size: {original_size}")
    
    # 加载模型和处理器
    print(f"Loading model from: {model_path}")
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForDepthEstimation.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # 预处理图片
    print("Preprocessing image...")
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推理
    print("Running depth estimation...")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 插值到原始尺寸
    print("Interpolating to original size...")
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=original_size[::-1],  # PIL size is (width, height), but tensor needs (height, width)
        mode="bicubic",
        align_corners=False,
    )
    
    # 转换为 numpy 数组并归一化
    depth = prediction.squeeze().cpu().numpy()
    
    # 归一化到 0-255 范围用于可视化
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # 保存深度图
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_depth.png"
    
    print(f"Saving depth map to: {output_path}")
    depth_image = Image.fromarray(depth_uint8, mode='L')
    depth_image.save(output_path)
    
    # 也可以保存彩色深度图（使用 colormap）
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # 创建彩色深度图
        depth_colored = cm.viridis(depth_normalized)[:, :, :3]  # 使用 viridis colormap
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_image = Image.fromarray(depth_colored)
        
        colored_output_path = output_path.replace('.png', '_colored.png')
        depth_colored_image.save(colored_output_path)
        print(f"Saving colored depth map to: {colored_output_path}")
    except ImportError:
        print("matplotlib not available, skipping colored depth map")
    
    # 保存原始深度值（numpy 格式）
    depth_npy_path = output_path.replace('.png', '.npy')
    np.save(depth_npy_path, depth)
    print(f"Saving raw depth values to: {depth_npy_path}")
    
    print("Done!")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    
    return depth, output_path


def main():
    parser = argparse.ArgumentParser(description="使用 Depth-Anything-v2 生成深度图")
    parser.add_argument(
        "--image",
        type=str,
        default="/202421000505/wsh_project/refine_splatter/splatter-image_dpt/DPT_img/4.png",
        help="输入图片路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/202421000505/wsh_project/refine_splatter/splatter-image_dpt/scene/model_cache/Depth-Anything-v2",
        help="Depth-Anything-v2 模型路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/202421000505/wsh_project/refine_splatter/splatter-image_dpt/DPT_img/4_depth.png",
        help="输出深度图路径（可选）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="设备（cuda 或 cpu，默认自动选择）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model path not found: {args.model}")
        return
    
    # 生成深度图
    generate_depth_map(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()

