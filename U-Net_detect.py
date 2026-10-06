from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import glob
from tqdm import tqdm
import time


torch.set_num_threads(4)

# 卷积块
def conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1):
    block = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
    return block

# U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # 编码器 (每个卷积块后都有池化)
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        # 瓶颈层
        self.bottleneck = conv_block(256, 512)
        # 解码器 (上采样)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)  # 1024 = 512(跳跃连接) + 512(上采样)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)  # 512 = 256(跳跃连接) + 256(上采样)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)  # 256 = 128(跳跃连接) + 128(上采样)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)  # 128 = 64(跳跃连接) + 64(上采样)
        # 最终卷积
        self.final_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # 编码器路径 (下采样)
        e1 = self.enc1(x)  # 初始尺寸
        e2 = self.enc2(self.pool1(e1))  # 尺寸减半
        e3 = self.enc3(self.pool2(e2))  # 尺寸再减半
        e4 = self.enc4(self.pool3(e3))  # 尺寸再减半

        # 瓶颈
        b = self.bottleneck(self.pool4(e4))  # 最小尺寸

        # 解码器路径 (上采样 + 跳跃连接)
        d4 = self.up4(b)  # 上采样
        d4 = torch.cat([e4, d4], dim=1)  # 跳跃连接
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))

# 加载模型
def load_model(model_path, device='cpu'):
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"成功加载 U-Net 模型：{model_path}")
    return model

# 图像预处理
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    img = img / 255.0
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# 结果后处理
def postprocess_result(output_tensor, threshold=0.5):
    output = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (output > threshold).astype(np.uint8) * 255
    return binary_mask

# 单模型检测
def detect_images(model_path, detect_img_path, result_mask_path, device='cpu'):
    os.makedirs(result_mask_path, exist_ok=True)
    
    model = load_model(model_path, device)
    print(f"模型加载完成：{model_path}")
    
    img_ext = '.png'
    detect_imgs = sorted(Path(detect_img_path).glob(f'*{img_ext}'))
    print(f"找到 {len(detect_imgs)} 张图片需要检测")
    
    total_inference_time = 0
    inference_times = []
    
    with torch.no_grad():
        for img_path in tqdm(detect_imgs, desc="检测进度"):
            img_tensor = preprocess_image(img_path).to(device)
            
            start_time = time.time()
            output = model(img_tensor)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            total_inference_time += inference_time
            
            binary_mask = postprocess_result(output)
            
            result_file_name = img_path.name
            result_file_path = os.path.join(result_mask_path, result_file_name)
            cv2.imwrite(str(result_file_path), binary_mask)
    
    avg_inference_time = total_inference_time / len(detect_imgs) if detect_imgs else 0
    
    stats_file_path = os.path.join(result_mask_path, 'inference_stats.txt')
    with open(stats_file_path, 'w', encoding='utf-8') as f:
        f.write(f"总图片数：{len(detect_imgs)}\n")
        f.write(f"总推理耗时：{total_inference_time:.4f} 秒\n")
        f.write(f"平均每张图片推理耗时：{avg_inference_time*1000:.2f} 毫秒\n")
    
    print(f"所有检测完成，二值掩膜保存至：{result_mask_path}")
    print(f"推理统计信息已保存至：{stats_file_path}")
    return detect_imgs

# 高斯模糊背景虚化
def background_blur_gaussian(image_path, mask_path, output_path, blur_strength):
    """
    使用高斯模糊实现背景虚化
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    foreground = image.copy()
    background = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

    mask_3ch = cv2.merge([mask, mask, mask])
    mask_normalized = mask_3ch.astype(float) / 255.0
    result = foreground * mask_normalized + background * (1 - mask_normalized)
    result = result.astype(np.uint8)

    cv2.imwrite(output_path, result)
    return result

# 批量背景虚化
def batch_background_blur_gaussian(images_dir, masks_folder, output_folder, blur_strength=20):
    """
    对整个文件夹的图像进行批量高斯模糊背景虚化
    """
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    print(f"找到 {len(image_files)} 个图像文件用于背景虚化")

    success_count = 0
    for image_path in tqdm(image_files, desc="背景虚化进度"):
        filename = Path(image_path).stem
        image_ext = Path(image_path).suffix

        mask_filename = f"{filename}.png"
        mask_path = os.path.join(masks_folder, mask_filename)

        if not os.path.exists(mask_path):
            print(f"跳过 {filename}: 掩码文件 {mask_filename} 不存在")
            continue

        output_filename = f"{filename}_blurred{image_ext}"
        output_path = os.path.join(output_folder, output_filename)

        background_blur_gaussian(image_path, mask_path, output_path, blur_strength)
        success_count += 1

    print(f"\n批量背景虚化完成！成功处理 {success_count}/{len(image_files)} 个图像")
    return success_count


model_path = r"C:\Users\23005\Desktop\result\contrast1_U-Net_result\unet_model.pth"
detect_img_path = r"./datasets2/img/detect"
result_mask_path = r"./datasets2/U-Net_mask"
result_path = r"./datasets2/U-Net_gaussian"

detect_images(model_path, detect_img_path, result_mask_path, 'cuda')
batch_background_blur_gaussian(detect_img_path, result_mask_path, result_path, blur_strength=115)

print(f"\n所有任务完成！")
print(f"- 检测结果（二值掩膜）保存至：{result_mask_path}")
print(f"- 背景虚化结果保存至：{result_path}")


