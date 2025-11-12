# objaverse.py
import os
import glob
import json
import math
import torch
import torchvision
import numpy as np

from PIL import Image

from .shared_dataset import SharedDataset  # 继承共享数据集基类

from utils.graphics_utils import getProjectionMatrix, fov2focal  # 图形学工具函数
from utils.camera_utils import get_loop_cameras  # 相机工具函数

# 数据集路径配置（需要用户修改）
OBJAVERSE_ROOT = "/202421000505/wsh_project/Dataset_Objaverse/views_release/"  # 修改为你的数据目录路径
OBJAVERSE_LVIS_ANNOTATION_PATH = "/202421000505/wsh_project/Dataset_Objaverse/200GB_Objaverse.json"  # 修改为过滤用的.json文件路径
# OBJAVERSE_ROOT = "/202421000505/wsh_project/refine_splatter/Objaverse/views_release/"  # 修改为你的数据目录路径
# OBJAVERSE_LVIS_ANNOTATION_PATH = "/202421000505/wsh_project/refine_splatter/Objaverse/mini.json"  # 修改为过滤用的.json文件路径
assert OBJAVERSE_ROOT is not None, "Update dataset path"
assert OBJAVERSE_LVIS_ANNOTATION_PATH is not None, "Update filtering .json path"

class ObjaverseDataset(SharedDataset):
    """Objaverse数据集类，用于处理3D对象的图像和相机数据"""
    
    def __init__(self,
                 cfg,  # 配置文件
                 dataset_name="train"  # 数据集类型：train/val/vis/test
                 ) -> None:

        super(ObjaverseDataset).__init__()
        self.cfg = cfg
        self.root_dir = OBJAVERSE_ROOT  # 数据集根目录

        # 加载文件路径列表
        with open(OBJAVERSE_LVIS_ANNOTATION_PATH) as f:
            self.paths = json.load(f)

        # 划分训练集和验证集
        total_objects = len(self.paths)
        self.dataset_name = dataset_name
        
        if self.dataset_name == "val" or dataset_name == "vis":
            # 验证集或可视化集：使用最后1%作为验证
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]
        elif self.dataset_name == "test":
            raise NotImplementedError  # Objaverse没有单独的测试子集
        else:
            # 训练集：使用前99%作为训练
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]
        
        # 如果配置了子集大小，进行截取
        if cfg.data.subset != -1:
            self.paths = self.paths[:cfg.data.subset]

        print('============= length of dataset %d =============' % len(self.paths))

        # 计算投影矩阵（用于3D到2D的投影）
        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,  # 近远裁剪平面
            fovX=cfg.data.fov * 2 * np.pi / 360,  # 水平视野（弧度）
            fovY=cfg.data.fov * 2 * np.pi / 360   # 垂直视野（弧度）
        ).transpose(0,1)

        self.image_side_target = self.cfg.data.training_resolution  # 目标图像分辨率
        
        # OpenGL到COLMAP坐标系的转换矩阵
        # OpenGL: x右, y上, z向相机内（向后）
        # COLMAP/OpenCV: x右, y下, z远离相机（向前）
        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)

        self.imgs_per_obj_train = self.cfg.opt.imgs_per_obj  # 每个对象使用的图像数量

    def __len__(self):
        """返回数据集中对象的数量"""
        return len(self.paths)
       
    def load_imgs_and_convert_cameras(self, paths, num_views):
        """
        加载图像、相机矩阵和投影矩阵
        
        参数:
            paths: 图像文件路径列表
            num_views: 要加载的视角数量
            
        返回:
            包含图像和相机数据的字典
        """
        # 设置背景颜色为白色
        bg_color = torch.tensor([1., 1., 1.], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        
        # 初始化列表存储各种变换矩阵和图像数据
        world_view_transforms = []  # 世界到相机视图的变换矩阵
        view_world_transforms = []  # 相机视图到世界的变换矩阵
        camera_centers = []         # 相机中心位置
        imgs = []                   # 图像数据
        fg_masks = []               # 前景掩码

        # 确定要加载的图像索引
        if self.dataset_name != "train":
            # 验证集：使用固定顺序确保可重复性
            indexes = torch.arange(num_views)
        else:
            # 训练集：随机采样
            indexes = torch.randperm(len(paths))[:num_views]
            # 确保前input_images张作为条件图像
            indexes = torch.cat([indexes[:self.cfg.data.input_images], indexes], dim=0)

        # 加载图像和相机数据
        for i in indexes:
            # 读取图像并转换为[0,1]范围的FloatTensor，调整到训练分辨率
            img = Image.open(paths[i])
            # 调整图像大小
            img = torchvision.transforms.functional.resize(
                img, self.cfg.data.training_resolution,
                interpolation=torchvision.transforms.InterpolationMode.LANCZOS
            )
            img = torchvision.transforms.functional.pil_to_tensor(img) / 255.0
            
            # 处理前景和背景：RGBA图像的A通道作为前景掩码
            fg_masks.append(img[3:, ...])  # Alpha通道作为前景掩码
            # 应用前景掩码：前景保持原色，背景设为白色
            imgs.append(img[:3, ...] * img[3:, ...] + bg_color * (1 - img[3:, ...]))

            # 加载相机位姿数据（.npy文件存储世界到相机的列主序矩阵）
            w2c_cmo = torch.tensor(np.load(paths[i].replace('png', 'npy'))).float()  # 3x4
            w2c_cmo = torch.cat([w2c_cmo, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)  # 扩展为4x4
            
            # 坐标系转换：从OpenGL转换为COLMAP/OpenCV
            w2c_cmo = torch.matmul(self.opengl_to_colmap, w2c_cmo)
            
            # 转换为行主序（PyTorch/TensorFlow常用格式）
            world_view_transform = w2c_cmo.transpose(0, 1)        # 世界到视图变换
            view_world_transform = w2c_cmo.inverse().transpose(0, 1)  # 视图到世界变换
            camera_center = view_world_transform[3, :3].clone()   # 相机中心位置

            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)
            camera_centers.append(camera_center)

        # 将列表转换为张量
        imgs = torch.stack(imgs)
        fg_masks = torch.stack(fg_masks)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)
        
        # 计算焦距（从视野角度转换）
        focals_pixels = torch.full((imgs.shape[0], 2),
                                   fill_value=fov2focal(self.cfg.data.fov,
                                                        self.cfg.data.training_resolution))
        pps_pixels = torch.zeros((imgs.shape[0], 2))  # 主点坐标（通常为图像中心）

        # 调整相机距离的缩放因子，确保第一个相机到原点的距离为2.0
        assert torch.norm(camera_centers[0]) > 1e-5, \
            "Camera is at {} from center".format(torch.norm(camera_centers[0]))
        translation_scaling_factor = 2.0 / torch.norm(camera_centers[0])
        world_view_transforms[:, 3, :3] *= translation_scaling_factor
        view_world_transforms[:, 3, :3] *= translation_scaling_factor
        camera_centers *= translation_scaling_factor

        # 计算完整的投影变换矩阵（视图变换 × 投影矩阵）
        full_proj_transforms = world_view_transforms.bmm(
            self.projection_matrix.unsqueeze(0).expand(world_view_transforms.shape[0], 4, 4)
        )

        return {
            "gt_images": imgs,                    # 真实图像
            "world_view_transforms": world_view_transforms,      # 世界到视图变换
            "view_to_world_transforms": view_world_transforms,   # 视图到世界变换
            "full_proj_transforms": full_proj_transforms,        # 完整投影变换
            "camera_centers": camera_centers,     # 相机中心位置
            "focals_pixels": focals_pixels,       # 焦距（像素单位）
            "pps_pixels": pps_pixels,             # 主点坐标
            "fg_masks": fg_masks                  # 前景掩码
        }

    def load_loop(self, paths, num_imgs_in_loop):
        """
        加载环形相机路径的数据，用于可视化
        
        参数:
            paths: 图像路径列表
            num_imgs_in_loop: 环形路径中的图像数量
            
        返回:
            包含真实图像和环形相机数据的字典
        """
        world_view_transforms = []
        view_world_transforms = []
        camera_centers = []
        imgs = []

        # 首先加载所有真实图像和相机数据
        gt_imgs_and_cameras = self.load_imgs_and_convert_cameras(paths, len(paths))
        
        # 生成环形相机路径
        loop_cameras_c2w_cmo = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop)

        # 添加条件图像（输入图像）
        for src_idx in range(self.cfg.data.input_images):
            imgs.append(gt_imgs_and_cameras["gt_images"][src_idx])
            camera_centers.append(gt_imgs_and_cameras["camera_centers"][src_idx])
            world_view_transforms.append(gt_imgs_and_cameras["world_view_transforms"][src_idx])
            view_world_transforms.append(gt_imgs_and_cameras["view_to_world_transforms"][src_idx])

        # 添加环形路径中的虚拟相机
        for loop_camera_c2w_cmo in loop_cameras_c2w_cmo:
            # 计算变换矩阵
            view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1)
            world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()

            camera_centers.append(camera_center)
            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)

            # 为虚拟相机找到最接近的真实图像作为参考
            closest_gt_idx = torch.argmin(torch.norm(
                gt_imgs_and_cameras["camera_centers"] - camera_center.unsqueeze(0), dim=-1
            )).item()
            imgs.append(gt_imgs_and_cameras["gt_images"][closest_gt_idx])

        # 转换为张量
        imgs = torch.stack(imgs)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)

        # 计算投影变换
        full_proj_transforms = world_view_transforms.bmm(
            self.projection_matrix.unsqueeze(0).expand(world_view_transforms.shape[0], 4, 4)
        )

        # 焦距和主点
        focals_pixels = torch.full((imgs.shape[0], 2),
                                   fill_value=fov2focal(self.cfg.data.fov,
                                                        self.cfg.data.training_resolution))
        pps_pixels = torch.zeros((imgs.shape[0], 2))

        return {
            "gt_images": imgs.to(memory_format=torch.channels_last),  # 使用通道最后格式优化内存
            "world_view_transforms": world_view_transforms,
            "view_to_world_transforms": view_world_transforms,
            "full_proj_transforms": full_proj_transforms,
            "camera_centers": camera_centers,
            "focals_pixels": focals_pixels,
            "pps_pixels": pps_pixels
        }

    def get_example_id(self, index):
        """获取示例的唯一标识符"""
        example_id = self.paths[index]
        return example_id

    def __getitem__(self, index):
        """获取单个数据样本"""
        # 构建文件路径
        filename = os.path.join(self.root_dir, self.paths[index])
        paths = glob.glob(filename + '/*.png')  # 获取所有PNG图像文件

        # 根据数据集类型选择加载方式
        if self.dataset_name == "vis":
            # 可视化模式：加载环形相机路径
            images_and_camera_poses = self.load_loop(paths, 200)
        else:
            # 训练或验证模式
            if self.dataset_name == "train":
                num_views = self.imgs_per_obj_train  # 训练时使用固定数量的视图
            else:
                num_views = len(paths)  # 验证时使用所有视图
            
            try:
                images_and_camera_poses = self.load_imgs_and_convert_cameras(paths, num_views)
            except:
                # 如果加载失败，使用备用对象
                print("Found an error with path {}, loading from \
                      8e348d4d2f2949cf88bd896a92a4364d instead").format(self.paths[index])
                filename = os.path.join(self.root_dir, '8e348d4d2f2949cf88bd896a92a4364d')
                paths = glob.glob(filename + '/*.png')
                num_views = len(paths)
                images_and_camera_poses = self.load_imgs_and_convert_cameras(paths, num_views)

        # 将相机位姿转换为相对于第一个相机的相对位姿
        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        # 获取源相机的四元数表示（从SharedDataset继承的方法）
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(
            images_and_camera_poses["view_to_world_transforms"]
        )

        return images_and_camera_poses