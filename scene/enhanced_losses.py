import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAwareLoss(nn.Module):
    """深度感知损失函数，用于优化DPT融合"""
    
    def __init__(self, alpha=0.1, beta=0.05):
        super().__init__()
        self.alpha = alpha  # 深度一致性权重
        self.beta = beta    # 特征平滑性权重
        
    def forward(self, pred_depth, gt_depth=None, features=None):
        """
        计算深度感知损失
        Args:
            pred_depth: 预测深度 [B, 1, H, W]
            gt_depth: 真实深度 [B, 1, H, W] (可选)
            features: 特征图 [B, C, H, W] (可选)
        Returns:
            loss: 深度感知损失
        """
        total_loss = 0.0
        
        # 深度平滑性损失
        if pred_depth is not None:
            # 计算深度梯度
            depth_grad_x = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])
            depth_grad_y = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])
            
            smooth_loss = torch.mean(depth_grad_x) + torch.mean(depth_grad_y)
            total_loss += self.alpha * smooth_loss
        
        # 特征一致性损失
        if features is not None:
            # 计算特征的空间一致性
            feat_grad_x = torch.mean(torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1]), dim=1)
            feat_grad_y = torch.mean(torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :]), dim=1)
            
            consistency_loss = torch.mean(feat_grad_x) + torch.mean(feat_grad_y)
            total_loss += self.beta * consistency_loss
        
        return total_loss


class FeatureConsistencyLoss(nn.Module):
    """特征一致性损失函数"""
    
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
        
    def forward(self, unet_features, dpt_features):
        """
        计算特征一致性损失
        Args:
            unet_features: UNet特征 [B, C, H, W]
            dpt_features: DPT特征 [B, C', H', W']
        Returns:
            loss: 特征一致性损失
        """
        # 确保特征尺寸一致
        if unet_features.shape != dpt_features.shape:
            # 调整DPT特征到UNet特征的尺寸
            dpt_features = F.interpolate(dpt_features, size=unet_features.shape[-2:], 
                                       mode='bilinear', align_corners=False)
            
            # 如果通道数不匹配，使用平均池化来调整通道数
            if dpt_features.shape[1] != unet_features.shape[1]:
                if dpt_features.shape[1] > unet_features.shape[1]:
                    # DPT特征通道数更多，使用1x1卷积降维
                    if not hasattr(self, 'dpt_projection'):
                        self.dpt_projection = nn.Conv2d(dpt_features.shape[1], unet_features.shape[1], 
                                                      kernel_size=1).to(dpt_features.device)
                    dpt_features = self.dpt_projection(dpt_features)
                else:
                    # DPT特征通道数更少，重复通道
                    repeat_factor = unet_features.shape[1] // dpt_features.shape[1]
                    remainder = unet_features.shape[1] % dpt_features.shape[1]
                    dpt_features = dpt_features.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        padding = dpt_features[:, :remainder, :, :]
                        dpt_features = torch.cat([dpt_features, padding], dim=1)
        
        # 计算特征相似性
        similarity = F.cosine_similarity(unet_features, dpt_features, dim=1)
        consistency_loss = 1.0 - torch.mean(similarity)
        
        return self.weight * consistency_loss


class AdaptiveFusionLoss(nn.Module):
    """自适应融合损失函数"""
    
    def __init__(self, fusion_weight=0.1):
        super().__init__()
        self.fusion_weight = fusion_weight
        
    def forward(self, fused_features, unet_features, dpt_features):
        """
        计算自适应融合损失
        Args:
            fused_features: 融合后特征 [B, C, H, W]
            unet_features: UNet特征 [B, C, H, W]
            dpt_features: DPT特征 [B, C', H', W']
        Returns:
            loss: 自适应融合损失
        """
        # 确保特征尺寸一致
        if dpt_features.shape != unet_features.shape:
            # 调整DPT特征到UNet特征的尺寸
            dpt_features = F.interpolate(dpt_features, size=unet_features.shape[-2:], 
                                       mode='bilinear', align_corners=False)
            
            # 如果通道数不匹配，调整通道数
            if dpt_features.shape[1] != unet_features.shape[1]:
                if dpt_features.shape[1] > unet_features.shape[1]:
                    # DPT特征通道数更多，使用1x1卷积降维
                    if not hasattr(self, 'dpt_projection'):
                        self.dpt_projection = nn.Conv2d(dpt_features.shape[1], unet_features.shape[1], 
                                                      kernel_size=1).to(dpt_features.device)
                    dpt_features = self.dpt_projection(dpt_features)
                else:
                    # DPT特征通道数更少，重复通道
                    repeat_factor = unet_features.shape[1] // dpt_features.shape[1]
                    remainder = unet_features.shape[1] % dpt_features.shape[1]
                    dpt_features = dpt_features.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        padding = dpt_features[:, :remainder, :, :]
                        dpt_features = torch.cat([dpt_features, padding], dim=1)
        
        # 计算融合质量
        unet_sim = F.cosine_similarity(fused_features, unet_features, dim=1)
        dpt_sim = F.cosine_similarity(fused_features, dpt_features, dim=1)
        
        # 融合损失：融合特征应该同时保持UNet和DPT的特征
        fusion_loss = (1.0 - torch.mean(unet_sim)) + (1.0 - torch.mean(dpt_sim))
        
        return self.fusion_weight * fusion_loss


class EnhancedDPTLoss(nn.Module):
    """增强的DPT损失函数组合"""
    
    def __init__(self, depth_weight=0.1, consistency_weight=0.05, fusion_weight=0.1):
        super().__init__()
        self.depth_loss = DepthAwareLoss(alpha=depth_weight, beta=consistency_weight)
        self.consistency_loss = FeatureConsistencyLoss(weight=consistency_weight)
        self.fusion_loss = AdaptiveFusionLoss(fusion_weight=fusion_weight)
        
    def forward(self, pred_depth=None, gt_depth=None, unet_features=None, 
                dpt_features=None, fused_features=None):
        """
        计算增强的DPT损失
        Args:
            pred_depth: 预测深度 [B, 1, H, W]
            gt_depth: 真实深度 [B, 1, H, W]
            unet_features: UNet特征 [B, C, H, W]
            dpt_features: DPT特征 [B, C, H, W]
            fused_features: 融合特征 [B, C, H, W]
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 深度感知损失
        if pred_depth is not None:
            depth_loss = self.depth_loss(pred_depth, gt_depth, unet_features)
            total_loss += depth_loss
            loss_dict['depth_loss'] = depth_loss.item()
        
        # 特征一致性损失
        if unet_features is not None and dpt_features is not None:
            consistency_loss = self.consistency_loss(unet_features, dpt_features)
            total_loss += consistency_loss
            loss_dict['consistency_loss'] = consistency_loss.item()
        
        # 自适应融合损失
        if fused_features is not None and unet_features is not None and dpt_features is not None:
            fusion_loss = self.fusion_loss(fused_features, unet_features, dpt_features)
            total_loss += fusion_loss
            loss_dict['fusion_loss'] = fusion_loss.item()
        
        return total_loss, loss_dict
