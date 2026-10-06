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

# SE 注意力机制模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Attention Gate 注意力门模块
class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.bn_g = nn.BatchNorm2d(inter_channels)
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.bn_x = nn.BatchNorm2d(inter_channels)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False)
        self.bn_psi = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        g1 = self.bn_g(g1)
        x1 = self.W_x(x)
        x1 = self.bn_x(x1)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.bn_psi(psi)
        psi = self.sigmoid(psi)
        return x * psi

# 基础 ResidualBlock（无注意力机制，用于 ResNet_train1.py）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 带 SE 注意力机制的 ResidualBlock（用于 ResNet_train.py 和 ResNet_train3.py）
class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, se_reduction=16):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.se_block = SEBlock(out_channels, se_reduction)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se_block(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet50 Bottleneck Block
class BottleneckBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

# ResNet 模型基类（用于 ResNet_train1.py - 无注意力机制）
class ResNetBasic(nn.Module):
    def __init__(self, block, layers, in_channels=3, out_channels=1):
        super(ResNetBasic, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.up4_conv1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.up4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.up3_conv1x1 = nn.Conv2d(256, 128, kernel_size=1)
        self.up3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.up2_conv1x1 = nn.Conv2d(128, 64, kernel_size=1)
        self.up2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up1_conv1x1 = nn.Conv2d(64, 64, kernel_size=1)
        self.up1_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up0_conv1x1 = nn.Conv2d(64, 32, kernel_size=1)
        self.up0_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    def _make_layer(self, block, out_channels, blocks, stride=1, in_channels=None):
        current_in_channels = in_channels if in_channels is not None else self.in_channels
        downsample = None
        if stride != 1 or current_in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(current_in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(current_in_channels, out_channels, stride, downsample)]
        if in_channels is None:
            self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2, pool_indices = self.maxpool(x1)
        
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        
        out = nn.functional.interpolate(x6, size=x5.shape[2:], mode='nearest')
        out = self.up4_conv1x1(out)
        out = self.up4_conv(out)
        
        out = nn.functional.interpolate(out, size=x4.shape[2:], mode='nearest')
        out = self.up3_conv1x1(out)
        out = self.up3_conv(out)
        
        out = nn.functional.interpolate(out, size=x3.shape[2:], mode='nearest')
        out = self.up2_conv1x1(out)
        out = self.up2_conv(out)
        
        out = nn.functional.interpolate(out, size=x1.shape[2:], mode='nearest')
        out = self.up1_conv1x1(out)
        out = self.up1_conv(out)
        
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='nearest')
        out = self.up0_conv1x1(out)
        out = self.up0_conv(out)
        
        return torch.sigmoid(self.final_conv(out))

# ResNet 模型类（支持 BottleneckBlock，用于 ResNet50）
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, out_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.up4_conv1x1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.up4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.up3_conv1x1 = nn.Conv2d(256, 128, kernel_size=1)
        self.up3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.up2_conv1x1 = nn.Conv2d(128, 64, kernel_size=1)
        self.up2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up1_conv1x1 = nn.Conv2d(64, 64, kernel_size=1)
        self.up1_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up0_conv1x1 = nn.Conv2d(64, 32, kernel_size=1)
        self.up0_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    def _make_layer(self, block, out_channels, blocks, stride=1, in_channels=None):
        current_in_channels = in_channels if in_channels is not None else self.in_channels
        downsample = None
        if stride != 1 or current_in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(current_in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(current_in_channels, out_channels, stride, downsample)]
        if in_channels is None:
            self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2, pool_indices = self.maxpool(x1)
        
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        
        out = nn.functional.interpolate(x6, size=x5.shape[2:], mode='nearest')
        out = self.up4_conv1x1(out)
        out = self.up4_conv(out)
        
        out = nn.functional.interpolate(out, size=x4.shape[2:], mode='nearest')
        out = self.up3_conv1x1(out)
        out = self.up3_conv(out)
        
        out = nn.functional.interpolate(out, size=x3.shape[2:], mode='nearest')
        out = self.up2_conv1x1(out)
        out = self.up2_conv(out)
        
        out = nn.functional.interpolate(out, size=x1.shape[2:], mode='nearest')
        out = self.up1_conv1x1(out)
        out = self.up1_conv(out)
        
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='nearest')
        out = self.up0_conv1x1(out)
        out = self.up0_conv(out)
        
        return torch.sigmoid(self.final_conv(out))

# 带 Attention Gate 的 ResNet 模型（用于 ResNet_train.py 和 ResNet_train2.py）
class ResNetAttention(nn.Module):
    def __init__(self, block, layers, in_channels=3, out_channels=1):
        super(ResNetAttention, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.attn_gate4 = AttentionGate(g_channels=256, x_channels=256, inter_channels=256)
        self.attn_gate3 = AttentionGate(g_channels=128, x_channels=128, inter_channels=128)
        self.attn_gate2 = AttentionGate(g_channels=64, x_channels=64, inter_channels=64)
        self.attn_gate1 = AttentionGate(g_channels=64, x_channels=64, inter_channels=64)
        
        self.up4_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up4_conv1x1 = nn.Conv2d(512, 256, kernel_size=1)
        
        self.up3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up3_conv1x1 = nn.Conv2d(256, 128, kernel_size=1)
        
        self.up2_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up2_conv1x1 = nn.Conv2d(128, 64, kernel_size=1)
        
        self.up1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up1_conv1x1 = nn.Conv2d(64, 64, kernel_size=1)
        
        self.up0_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.up0_conv1x1 = nn.Conv2d(64, 32, kernel_size=1)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    def _make_layer(self, block, out_channels, blocks, stride=1, in_channels=None):
        current_in_channels = in_channels if in_channels is not None else self.in_channels
        downsample = None
        if stride != 1 or current_in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(current_in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(current_in_channels, out_channels, stride, downsample)]
        if in_channels is None:
            self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2, pool_indices = self.maxpool(x1)
        
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        
        out = nn.functional.interpolate(x6, size=x5.shape[2:], mode='nearest')
        out = self.up4_conv1x1(out)
        attn_x5 = self.attn_gate4(out, x5)
        out = torch.cat([attn_x5, out], dim=1)
        out = self.up4_conv(out)
        
        out = nn.functional.interpolate(out, size=x4.shape[2:], mode='nearest')
        out = self.up3_conv1x1(out)
        attn_x4 = self.attn_gate3(out, x4)
        out = torch.cat([attn_x4, out], dim=1)
        out = self.up3_conv(out)
        
        out = nn.functional.interpolate(out, size=x3.shape[2:], mode='nearest')
        out = self.up2_conv1x1(out)
        attn_x3 = self.attn_gate2(out, x3)
        out = torch.cat([attn_x3, out], dim=1)
        out = self.up2_conv(out)
        
        out = nn.functional.interpolate(out, size=x1.shape[2:], mode='nearest')
        out = self.up1_conv1x1(out)
        attn_x1 = self.attn_gate1(out, x1)
        out = torch.cat([attn_x1, out], dim=1)
        out = self.up1_conv(out)
        
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='nearest')
        out = self.up0_conv1x1(out)
        out = self.up0_conv(out)
        
        return torch.sigmoid(self.final_conv(out))

# 模型创建函数
def resnet18_basic(in_channels=3, out_channels=1):
    return ResNetBasic(ResidualBlock, [2, 2, 2, 2], in_channels, out_channels)

def resnet18_se(in_channels=3, out_channels=1):
    return ResNetBasic(ResidualBlockSE, [2, 2, 2, 2], in_channels, out_channels)

def resnet18_attention(in_channels=3, out_channels=1):
    return ResNetAttention(ResidualBlock, [2, 2, 2, 2], in_channels, out_channels)

def resnet18_se_attention(in_channels=3, out_channels=1):
    return ResNetAttention(ResidualBlockSE, [2, 2, 2, 2], in_channels, out_channels)

def resnet34(in_channels=3, out_channels=1):
    return ResNetBasic(ResidualBlock, [3, 4, 6, 3], in_channels, out_channels)

def resnet50(in_channels=3, out_channels=1):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], in_channels, out_channels)

# 加载模型（自动检测模型类型）
def load_model(model_path, device='cpu'):
    model_configs = [
        ('basic', resnet18_basic),
        ('se', resnet18_se),
        ('attention', resnet18_attention),
        ('se_attention', resnet18_se_attention),
        ('resnet34', resnet34),
        ('resnet50', resnet50)
    ]
    
    for model_name, model_fn in model_configs:
        try:
            model = model_fn(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"成功加载模型类型：{model_name}")
            return model
        except Exception as e:
            continue
    
    raise RuntimeError(f"无法加载模型 {model_path}，可能是不兼容的模型架构")

'''Preprocess Image'''
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    img = img / 255.0
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor

'''Postprocess Result'''
def postprocess_result(output_tensor, threshold=0.5):
    output = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (output > threshold).astype(np.uint8) * 255
    return binary_mask

'''Detect Images'''
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

# 单模型推理配置
model_path = r"C:\Users\23005\Desktop\result\contrast3_ResNet50_result\resnet50_model.pth"
detect_img_path = r"./datasets2/img/detect"
result_mask_path = r"./datasets2/ResNet50_mask"
result_path = r"./datasets2/ResNet50_gaussian"


detect_images(model_path, detect_img_path, result_mask_path, 'cuda')
batch_background_blur_gaussian(detect_img_path, result_mask_path, result_path, blur_strength=115)

print(f"\n所有任务完成！")
print(f"- 检测结果（二值掩膜）保存至：{result_mask_path}")
print(f"- 背景虚化结果保存至：{result_path}")
