import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DPTForDepthEstimation, DPTImageProcessor
import os


class DPTFeatureExtractor(nn.Module):
    """DPT特征提取器，使用预训练的DPT模型"""
    
    def __init__(self, model_path="/202421000505/wsh_project/refine_splatter/splatter-image_dpt/scene/model_cache/dpt-large/"):
        super().__init__()
        # 预设特征维度
        self.feature_dim = 1024
        
        # 加载DPT模型
        try:
            if os.path.exists(model_path):
                print(f"Loading DPT model from {model_path}")
                self.dpt_model = DPTForDepthEstimation.from_pretrained(model_path)
                self.processor = DPTImageProcessor.from_pretrained(model_path)
            else:
                print(f"DPT model path {model_path} not found, loading from HuggingFace")
                self.dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
                self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        except Exception as e:
            msg = f"Failed to load DPT model: {e}"
            print(msg)
            raise RuntimeError(msg)
        
        # 冻结DPT模型的参数（可选）
        for param in self.dpt_model.parameters():
            param.requires_grad = False
        self.dpt_model.eval()
        
        # 特征后处理网络
        self.feature_postprocess = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1),
            nn.BatchNorm2d(self.feature_dim),
        )
        
        
    def forward(self, x):
        """
        提取DPT深度感知特征
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            features: DPT深度特征 [B, 1024, H', W']
        """
        if self.dpt_model is None:
            raise RuntimeError("DPT model not loaded. Cannot extract features.")
        # 使用真正的DPT模型提取特征
        with torch.no_grad():
            # 简化预处理：直接使用tensor输入
            # DPT模型期望输入在[-1, 1]范围内
            if x.max() > 1.0:  # 如果输入在[0, 1]范围
                x_normalized = x * 2.0 - 1.0
            else:
                x_normalized = x
            
            # 调整输入尺寸到DPT期望的尺寸
            if x_normalized.shape[-1] != 384 or x_normalized.shape[-2] != 384:
                x_resized = F.interpolate(x_normalized, size=(384, 384), mode='bilinear', align_corners=False)
            else:
                x_resized = x_normalized
            
            # 通过DPT模型获取特征
            outputs = self.dpt_model(x_resized, output_hidden_states=True)
            
            # 获取中间特征（通常是encoder的最后一层）
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # 使用最后一个隐藏状态作为特征
                hidden_state = outputs.hidden_states[-1]
                
                # 检查hidden_state的维度
                if len(hidden_state.shape) == 3:  # [B, seq_len, hidden_dim]
                    # 如果是1D序列，需要reshape为2D特征图
                    batch_size, seq_len, hidden_dim = hidden_state.shape
                    # 假设是正方形特征图
                    spatial_size = int(seq_len ** 0.5)
                    if spatial_size * spatial_size == seq_len:
                        features = hidden_state.view(batch_size, hidden_dim, spatial_size, spatial_size)
                    else:
                        # 如果不是完全平方数，使用自定义尺寸
                        features = hidden_state.view(batch_size, hidden_dim, -1, 1)
                        features = F.interpolate(features, size=(24, 24), mode='bilinear', align_corners=False)
                else:
                    features = hidden_state
            else:
                # 如果没有hidden_states，使用深度预测结果
                features = outputs.predicted_depth
                # 调整通道数到1024
                if features.shape[1] != self.feature_dim:
                    # 重复通道以匹配1024维
                    repeat_factor = self.feature_dim // features.shape[1]
                    features = features.repeat(1, repeat_factor, 1, 1)
                    if features.shape[1] < self.feature_dim:
                        # 如果还不够，用零填充
                        padding = torch.zeros(features.shape[0], self.feature_dim - features.shape[1], 
                                            features.shape[2], features.shape[3], device=features.device)
                        features = torch.cat([features, padding], dim=1)
            
            # 确保特征有正确的维度
            if len(features.shape) != 4:
                print(f"Warning: Unexpected feature shape {features.shape}, reshaping...")
                if len(features.shape) == 3:
                    batch_size, seq_len, hidden_dim = features.shape
                    spatial_size = int(seq_len ** 0.5)
                    if spatial_size * spatial_size == seq_len:
                        features = features.view(batch_size, hidden_dim, spatial_size, spatial_size)
                    else:
                        features = features.view(batch_size, hidden_dim, -1, 1)
                        features = F.interpolate(features, size=(24, 24), mode='bilinear', align_corners=False)
            
            # 调整特征尺寸到原始输入尺寸
            if features.shape[-2:] != x.shape[-2:]:
                features = F.interpolate(features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # 特征后处理
        features = self.feature_postprocess(features)
        
        return features


class DPTFeatureProjector(nn.Module):
    """增强的DPT特征投影器，使用更复杂的投影策略"""
    
    def __init__(self, dpt_feature_dim=1024, unet_feature_dim=512):
        super().__init__()
        
        # 使用更复杂的投影网络
        self.projection = nn.Sequential(
            # 第一层：降维
            nn.Conv2d(dpt_feature_dim, dpt_feature_dim // 2, kernel_size=1),
            nn.BatchNorm2d(dpt_feature_dim // 2),
            nn.ReLU(inplace=True),
            
            # 第二层：进一步降维
            nn.Conv2d(dpt_feature_dim // 2, dpt_feature_dim // 4, kernel_size=1),
            nn.BatchNorm2d(dpt_feature_dim // 4),
            nn.ReLU(inplace=True),
            
            # 第三层：最终投影
            nn.Conv2d(dpt_feature_dim // 4, unet_feature_dim, kernel_size=1),
            nn.BatchNorm2d(unet_feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 添加残差连接
        self.residual_proj = nn.Conv2d(dpt_feature_dim, unet_feature_dim, kernel_size=1) if dpt_feature_dim != unet_feature_dim else None
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, dpt_features):
        """
        将DPT特征投影到UNet特征空间
        Args:
            dpt_features: DPT特征 [B, 1024, H', W']
        Returns:
            projected_features: 投影后的特征 [B, unet_feature_dim, H', W']
        """
        # 主要投影路径
        projected = self.projection(dpt_features)
        
        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(dpt_features)
            projected = projected + residual
        
        return projected


class MultiScaleDPTFusion(nn.Module):
    """多尺度DPT特征与UNet特征融合模块"""
    
    def __init__(self, unet_feature_dim=512, dpt_model_path="/202421000505/wsh_project/refine_splatter/splatter-image/scene/model_cache/dpt-large"):
        super().__init__()
        
        # DPT特征提取器
        self.dpt_extractor = DPTFeatureExtractor(dpt_model_path)
        
        # 多尺度DPT特征投影器
        self.dpt_projectors = nn.ModuleList([
            DPTFeatureProjector(1024, unet_feature_dim // 4),  # 1/4尺度
            DPTFeatureProjector(1024, unet_feature_dim // 2),  # 1/2尺度
            DPTFeatureProjector(1024, unet_feature_dim),       # 全尺度
        ])
        
        # 多尺度特征融合网络
        self.scale_fusion = nn.ModuleList([
            # 1/4尺度融合
            nn.Sequential(
                nn.Conv2d(unet_feature_dim // 4 + unet_feature_dim // 4, unet_feature_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(unet_feature_dim // 4),
                nn.ReLU(inplace=True),
            ),
            # 1/2尺度融合
            nn.Sequential(
                nn.Conv2d(unet_feature_dim // 2 + unet_feature_dim // 2, unet_feature_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(unet_feature_dim // 2),
                nn.ReLU(inplace=True),
            ),
            # 全尺度融合
            nn.Sequential(
                nn.Conv2d(unet_feature_dim + unet_feature_dim, unet_feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(unet_feature_dim),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # 跨尺度特征融合
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv2d(unet_feature_dim * 3, unet_feature_dim, kernel_size=1),  # 3个尺度，每个都是unet_feature_dim
            nn.BatchNorm2d(unet_feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_feature_dim, unet_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(unet_feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 自适应权重学习
        self.adaptive_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(unet_feature_dim, unet_feature_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_feature_dim // 16, 3, kernel_size=1),  # 3个尺度的权重
            nn.Softmax(dim=1)
        )
        
        # 最终特征精炼
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(unet_feature_dim, unet_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(unet_feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_feature_dim, unet_feature_dim, kernel_size=1),
            nn.BatchNorm2d(unet_feature_dim),
        )
        
        # 预定义通道适配器
        self.unet_1_4_adapter = nn.Conv2d(unet_feature_dim, unet_feature_dim // 4, kernel_size=1)
        self.unet_1_2_adapter = nn.Conv2d(unet_feature_dim, unet_feature_dim // 2, kernel_size=1)
        self.unet_full_adapter = nn.Conv2d(unet_feature_dim, unet_feature_dim, kernel_size=1)
        
        # 跨尺度通道适配器
        self.channel_adapter_0 = nn.Conv2d(unet_feature_dim // 4, unet_feature_dim, kernel_size=1)
        self.channel_adapter_1 = nn.Conv2d(unet_feature_dim // 2, unet_feature_dim, kernel_size=1)
        self.channel_adapter_2 = nn.Conv2d(unet_feature_dim, unet_feature_dim, kernel_size=1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def to(self, device):
        """重写to方法，确保DPT模型也被移动到正确设备"""
        super().to(device)
        if hasattr(self, 'dpt_extractor') and hasattr(self.dpt_extractor, 'dpt_model'):
            self.dpt_extractor.dpt_model = self.dpt_extractor.dpt_model.to(device)
        return self
        
    def forward(self, unet_features, input_images):
        """
        多尺度DPT特征与UNet特征融合
        Args:
            unet_features: UNet encoder最后一层特征 [B, C, H, W]
            input_images: 原始输入图像 [B, 3, H, W]
        Returns:
            fused_features: 融合后的特征 [B, C, H, W]
        """
        # 提取DPT特征
        dpt_features = self.dpt_extractor(input_images)
        
        # 多尺度特征处理
        scale_features = []
        
        # 1/4尺度
        dpt_1_4 = self.dpt_projectors[0](dpt_features)
        unet_1_4 = F.avg_pool2d(unet_features, kernel_size=4, stride=4)
        if dpt_1_4.shape[-2:] != unet_1_4.shape[-2:]:
            dpt_1_4 = F.interpolate(dpt_1_4, size=unet_1_4.shape[-2:], mode='bilinear', align_corners=False)
        
        # 调整UNet特征通道数以匹配DPT特征
        unet_1_4_adjusted = F.adaptive_avg_pool2d(unet_features, (unet_1_4.shape[-2], unet_1_4.shape[-1]))
        unet_1_4_adjusted = self.unet_1_4_adapter(unet_1_4_adjusted)
        
        fused_1_4 = torch.cat([unet_1_4_adjusted, dpt_1_4], dim=1)
        fused_1_4 = self.scale_fusion[0](fused_1_4)
        scale_features.append(fused_1_4)
        
        # 1/2尺度
        dpt_1_2 = self.dpt_projectors[1](dpt_features)
        unet_1_2 = F.avg_pool2d(unet_features, kernel_size=2, stride=2)
        if dpt_1_2.shape[-2:] != unet_1_2.shape[-2:]:
            dpt_1_2 = F.interpolate(dpt_1_2, size=unet_1_2.shape[-2:], mode='bilinear', align_corners=False)
        
        # 调整UNet特征通道数以匹配DPT特征
        unet_1_2_adjusted = F.adaptive_avg_pool2d(unet_features, (unet_1_2.shape[-2], unet_1_2.shape[-1]))
        unet_1_2_adjusted = self.unet_1_2_adapter(unet_1_2_adjusted)
        
        fused_1_2 = torch.cat([unet_1_2_adjusted, dpt_1_2], dim=1)
        fused_1_2 = self.scale_fusion[1](fused_1_2)
        scale_features.append(fused_1_2)
        
        # 全尺度
        dpt_full = self.dpt_projectors[2](dpt_features)
        if dpt_full.shape[-2:] != unet_features.shape[-2:]:
            dpt_full = F.interpolate(dpt_full, size=unet_features.shape[-2:], mode='bilinear', align_corners=False)
        
        # 调整UNet特征通道数以匹配DPT特征
        unet_features_adjusted = self.unet_full_adapter(unet_features)
        
        fused_full = torch.cat([unet_features_adjusted, dpt_full], dim=1)
        fused_full = self.scale_fusion[2](fused_full)
        scale_features.append(fused_full)
        
        # 将不同尺度特征上采样到相同尺寸
        target_size = unet_features.shape[-2:]
        upsampled_features = []
        for i, feat in enumerate(scale_features):
            if feat.shape[-2:] != target_size:
                upsampled_feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            else:
                upsampled_feat = feat
            upsampled_features.append(upsampled_feat)
        
        # 跨尺度特征融合
        # 确保所有特征都有相同的通道数
        adjusted_features = []
        adjusted_features.append(self.channel_adapter_0(upsampled_features[0]))
        adjusted_features.append(self.channel_adapter_1(upsampled_features[1]))
        adjusted_features.append(self.channel_adapter_2(upsampled_features[2]))
        
        multi_scale_features = torch.cat(adjusted_features, dim=1)
        fused_features = self.cross_scale_fusion(multi_scale_features)
        
        # 自适应权重融合
        weights = self.adaptive_weights(fused_features)
        weighted_features = (fused_features * weights[:, 0:1, :, :] + 
                           adjusted_features[1] * weights[:, 1:2, :, :] + 
                           adjusted_features[2] * weights[:, 2:3, :, :])
        
        # 特征精炼
        refined_features = self.feature_refinement(weighted_features)
        
        # 残差连接
        final_features = refined_features + unet_features
        
        return final_features
