from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torchvision.transforms import functional as F
import random


torch.set_num_threads(4)

'''File Path'''
result_dir='./datasets2/melt2_ResNet_result'
train_img_path='./datasets2/img/train'
val_img_path='./datasets2/img/val'
train_mask_path='./datasets2/mask/train'
val_mask_path='./datasets2/mask/val'

'''Files Read'''
img_ext, mask_ext = '.png', '.png'
train_imgs = sorted(Path(train_img_path).glob(f'*{img_ext}'))
train_masks = sorted(Path(train_mask_path).glob(f'*{mask_ext}'))
val_imgs = sorted(Path(val_img_path).glob(f'*{img_ext}'))
val_masks = sorted(Path(val_mask_path).glob(f'*{mask_ext}'))

'''PNG 2 Array'''
train_img_list = []
train_mask_list = []
val_img_list = []
val_mask_list = []
for img_path in train_imgs:
    img = cv2.imread(str(img_path))
    train_img_list.append(img)
for mask_path in train_masks:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    train_mask_list.append(mask)
for img_path in val_imgs:
    img = cv2.imread(str(img_path))
    val_img_list.append(img)
for mask_path in val_masks:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    val_mask_list.append(mask)
train_imgs_array = np.array(train_img_list)
train_masks_array = np.array(train_mask_list)
val_imgs_array = np.array(val_img_list)
val_masks_array = np.array(val_mask_list)

'''Normalization 0-1'''
train_imgs_array = train_imgs_array/255.0
train_masks_array = train_masks_array/255.0
val_imgs_array = val_imgs_array/255.0
val_masks_array = val_masks_array/255.0

'''Array 2 Tensor'''
train_imgs_tensor = torch.from_numpy(train_imgs_array).float().permute(0, 3, 1, 2)
train_masks_tensor = torch.from_numpy(train_masks_array).float().unsqueeze(1)
val_imgs_tensor = torch.from_numpy(val_imgs_array).float().permute(0, 3, 1, 2)
val_masks_tensor = torch.from_numpy(val_masks_array).float().unsqueeze(1)

'''Data Loader'''
# 数据增强变换
class SegmentationTransforms:
    def __init__(self, rotation_range=10, scale_range=(0.9, 1.1), flip_prob=0.5, target_size=(512, 512)):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.target_size = target_size
    def __call__(self, image, mask):
        # 随机旋转
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        
        # 随机缩放 - 对于掩码使用最近邻插值保持二进制值
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.shape[1], image.shape[2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.resize(image, (new_h, new_w))
        mask = F.resize(mask, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)
        
        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        # 随机垂直翻转
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            mask = F.vflip(mask)
        
        # 固定大小调整
        image = F.resize(image, self.target_size)
        mask = F.resize(mask, self.target_size, interpolation=F.InterpolationMode.NEAREST)
        
        # 确保掩码值在0-1范围内
        mask = torch.clamp(mask, 0, 1)
        
        return image, mask
# 验证集变换（仅调整大小）
class ValTransforms:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
    def __call__(self, image, mask):
        # 固定大小调整
        image = F.resize(image, self.target_size)
        mask = F.resize(mask, self.target_size, interpolation=F.InterpolationMode.NEAREST)
        
        # 确保掩码值在0-1范围内
        mask = torch.clamp(mask, 0, 1)
        
        return image, mask
# 预加载数据到内存，避免重复读取
class PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images  # 已经加载到内存的tensor
        self.masks = masks
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
# 使用预加载的数据集，为训练集添加数据增强
train_transform = SegmentationTransforms(rotation_range=10, scale_range=(0.9, 1.1), flip_prob=0.5, target_size=(512, 512))
val_transform = ValTransforms(target_size=(512, 512))  # 验证集仅调整大小
train_dataset = PreloadedDataset(train_imgs_tensor, train_masks_tensor, transform=train_transform)
val_dataset = PreloadedDataset(val_imgs_tensor, val_masks_tensor, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

'''ResNet Block'''
# Attention Gate 注意力门模块
class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate, self).__init__()
        # 对解码器特征 g 的卷积变换
        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.bn_g = nn.BatchNorm2d(inter_channels)
        
        # 对编码器特征 x 的卷积变换
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.bn_x = nn.BatchNorm2d(inter_channels)
        
        # 注意力权重生成
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False)
        self.bn_psi = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        
        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        # 对 g 和 x 进行变换
        g1 = self.W_g(g)
        g1 = self.bn_g(g1)
        x1 = self.W_x(x)
        x1 = self.bn_x(x1)
        
        # 元素加法和激活
        psi = self.relu(g1 + x1)
        
        # 生成注意力权重
        psi = self.psi(psi)
        psi = self.bn_psi(psi)
        psi = self.sigmoid(psi)
        
        # 将注意力权重应用到 x 上
        return x * psi
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
'''ResNet for Segmentation with Unpooling'''
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, out_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 编码器
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)  # 保存池化索引
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 解码器 - 使用普通卷积替代转置卷积，结合池化索引上采样
        # 定义反池化层
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        
        # 注意力门 - 为每个跳跃连接点创建注意力门
        self.attn_gate4 = AttentionGate(g_channels=256, x_channels=256, inter_channels=256)  # 连接x5和上采样后的x6 (经过up4_conv1x1后变为256通道)
        self.attn_gate3 = AttentionGate(g_channels=128, x_channels=128, inter_channels=128)  # 连接x4和上采样后的输出 (经过up3_conv1x1后变为128通道)
        self.attn_gate2 = AttentionGate(g_channels=64, x_channels=64, inter_channels=64)   # 连接x3和上采样后的输出 (经过up2_conv1x1后变为64通道)
        self.attn_gate1 = AttentionGate(g_channels=64, x_channels=64, inter_channels=64)    # 连接x1和上采样后的输出 (经过up1_conv1x1后变为64通道)
        
        # 解码器卷积层 - 使用普通卷积替代残差块
        self.up4_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 输入通道数为512(拼接后)
        self.up4_conv1x1 = nn.Conv2d(512, 256, kernel_size=1)  # 1x1卷积调整通道数
        
        self.up3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 输入通道数为256(拼接后)
        self.up3_conv1x1 = nn.Conv2d(256, 128, kernel_size=1)  # 1x1卷积调整通道数
        
        self.up2_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 输入通道数为128(拼接后)
        self.up2_conv1x1 = nn.Conv2d(128, 64, kernel_size=1)  # 1x1卷积调整通道数
        
        self.up1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 输入通道数为128(拼接后)
        self.up1_conv1x1 = nn.Conv2d(64, 64, kernel_size=1)  # 1x1卷积调整通道数
        
        self.up0_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 输入通道数为32(经过1x1卷积后)
        self.up0_conv1x1 = nn.Conv2d(64, 32, kernel_size=1)  # 1x1卷积调整通道数
        
        # 最终卷积
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    def _make_layer(self, block, out_channels, blocks, stride=1, in_channels=None):
        # 如果提供了in_channels参数，则使用它，否则使用self.in_channels
        current_in_channels = in_channels if in_channels is not None else self.in_channels
        downsample = None
        if stride != 1 or current_in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(current_in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(current_in_channels, out_channels, stride, downsample)]
        # 只更新self.in_channels如果没有提供in_channels参数（即编码器部分）
        if in_channels is None:
            self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        # 编码器路径 - 保存池化索引
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2, pool_indices = self.maxpool(x1)  # 保存池化索引
        
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        
        # 解码器路径 - 使用普通卷积和池化索引上采样
        # 上采样x6到x5大小
        out = nn.functional.interpolate(x6, size=x5.shape[2:], mode='nearest')
        out = self.up4_conv1x1(out)
        # 使用注意力门 - g是解码器特征，x是编码器特征
        attn_x5 = self.attn_gate4(out, x5)
        out = torch.cat([attn_x5, out], dim=1)
        out = self.up4_conv(out)
        
        # 上采样到x4大小
        out = nn.functional.interpolate(out, size=x4.shape[2:], mode='nearest')
        out = self.up3_conv1x1(out)
        # 使用注意力门
        attn_x4 = self.attn_gate3(out, x4)
        out = torch.cat([attn_x4, out], dim=1)
        out = self.up3_conv(out)
        
        # 上采样到x3大小
        out = nn.functional.interpolate(out, size=x3.shape[2:], mode='nearest')
        out = self.up2_conv1x1(out)
        # 使用注意力门
        attn_x3 = self.attn_gate2(out, x3)
        out = torch.cat([attn_x3, out], dim=1)
        out = self.up2_conv(out)
        
        # 上采样到x1大小
        out = nn.functional.interpolate(out, size=x1.shape[2:], mode='nearest')
        out = self.up1_conv1x1(out)
        # 使用注意力门
        attn_x1 = self.attn_gate1(out, x1)
        out = torch.cat([attn_x1, out], dim=1)
        out = self.up1_conv(out)
        
        # 上采样到原始输入大小
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='nearest')
        out = self.up0_conv1x1(out)
        out = self.up0_conv(out)
        
        return torch.sigmoid(self.final_conv(out))
'''ResNet Model Creator'''
def resnet18(in_channels=3, out_channels=1):
    return ResNet(ResidualBlock, [2, 2, 2, 2], in_channels, out_channels)
# 计算IOU
def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()
# 计算mIOU
def calculate_miou(pred, target, threshold=0.5):
    (pred > threshold).float()
    iou_list = []
    for b in range(pred.shape[0]):
        iou = calculate_iou(pred[b], target[b], threshold)
        iou_list.append(iou)
    return np.mean(iou_list)
# 计算Dice系数
def calculate_dice(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    dice = (2. * intersection + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)
    return dice.item()

'''train'''
def resnet_train(train_loader, val_loader, in_channels=3, out_channels=1,
               epochs=20, learning_rate=0.01, device='cuda', patience=5, delta=0.0):

    model = resnet18(in_channels, out_channels).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # 初始化历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'iou': [],
        'miou': [],
        'dice': []
    }

    # 早停参数初始化
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = model.state_dict().copy()
    best_epoch = 0  # 记录最佳模型的轮数

    # 计算总图片数量用于进度条显示
    total_train_imgs = len(train_loader.dataset)
    total_val_imgs = len(val_loader.dataset)
    batch_size = train_loader.batch_size

    print("\n=== 开始训练 ===")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        processed_imgs = 0  # 已处理的图片数量

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            processed_imgs += data.size(0)  # 更新已处理的图片数量

            # 更新进度条，显示图片张数进度
            train_pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Imgs': f'{processed_imgs}/{total_train_imgs}'
            })
            train_pbar.update(1)

        train_pbar.close()
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        processed_val_imgs = 0  # 已处理的验证图片数量
        epoch_iou = 0
        epoch_miou = 0
        epoch_dice = 0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]', leave=False)

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
                processed_val_imgs += data.size(0)  # 更新已处理的验证图片数量
                
                # 计算IOU、mIOU和Dice系数
                iou = calculate_iou(output, target)
                miou = calculate_miou(output, target)
                dice = calculate_dice(output, target)
                
                epoch_iou += iou
                epoch_miou += miou
                epoch_dice += dice

                # 更新验证进度条，显示图片张数进度
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'IOU': f'{iou:.4f}',
                    'mIOU': f'{miou:.4f}',
                    'Dice': f'{dice:.4f}',
                    'Imgs': f'{processed_val_imgs}/{total_val_imgs}'
                })
                val_pbar.update(1)
        
        # 计算平均IOU、mIOU和Dice系数
        avg_iou = epoch_iou / len(val_loader)
        avg_miou = epoch_miou / len(val_loader)
        avg_dice = epoch_dice / len(val_loader)
        
        # 保存IOU、mIOU和Dice系数到历史记录
        history['iou'].append(avg_iou)
        history['miou'].append(avg_miou)
        history['dice'].append(avg_dice)

        val_pbar.close()
        avg_val_loss = epoch_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # 早停逻辑
        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # 更新最佳模型轮数
            patience_counter = 0
            # 保存最佳模型权重
            best_model_wts = model.state_dict().copy()
            print(f'Epoch {epoch + 1}/{epochs}: Saving best model')
        else:
            patience_counter += 1
            print(f'Epoch {epoch + 1}/{epochs}: Patience: {patience_counter}/{patience}')
        

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 检查是否触发早停条件
        if patience_counter >= patience:
            print(f"\n=== 早停触发! 验证损失在{patience}轮内没有改善 ===")
            break

    print("\n=== 训练完成 ===")

    # 保存训练结果 - 使用最佳模型
    print(f"\n=== 保存最佳模型 ===")
    print(f"正在加载轮次 {best_epoch} 的最佳模型权重...")
    model.load_state_dict(best_model_wts)
    
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    model_path = os.path.join(result_dir, 'resnet_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"已保存最佳模型到：{model_path}")
    print(f"最佳模型信息：轮次 {best_epoch}, 验证损失：{best_val_loss:.4f}")

    # 保存从第一轮到最佳验证轮次的数据
    if best_epoch > 0:  # 确保有有效的最佳模型轮次
        # 只保存到最佳轮次的数据
        train_loss_to_best = history['train_loss'][:best_epoch]
        val_loss_to_best = history['val_loss'][:best_epoch]
        iou_to_best = history['iou'][:best_epoch]
        miou_to_best = history['miou'][:best_epoch]
        dice_to_best = history['dice'][:best_epoch]
        
        # 保存损失数据
        np.save(os.path.join(result_dir, 'train_loss.npy'), train_loss_to_best)
        np.save(os.path.join(result_dir, 'val_loss.npy'), val_loss_to_best)
        
        # 保存IOU、mIOU和Dice数据
        np.save(os.path.join(result_dir, 'iou.npy'), iou_to_best)
        np.save(os.path.join(result_dir, 'miou.npy'), miou_to_best)
        np.save(os.path.join(result_dir, 'dice.npy'), dice_to_best)
    else:
        # 如果没有有效的最佳轮次，保存所有数据
        np.save(os.path.join(result_dir, 'train_loss.npy'), history['train_loss'])
        np.save(os.path.join(result_dir, 'val_loss.npy'), history['val_loss'])
        np.save(os.path.join(result_dir, 'iou.npy'), history['iou'])
        np.save(os.path.join(result_dir, 'miou.npy'), history['miou'])
        np.save(os.path.join(result_dir, 'dice.npy'), history['dice'])

    # 保存训练配置信息
    with open(os.path.join(result_dir, 'training_info.txt'), 'w') as f:
        f.write(f"Epochs: {epochs}\n")
        f.write(f"实际训练轮数: {epoch + 1}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Model Epoch: {best_epoch}\n")
        f.write(f"早停耐心值: {patience}\n")
        f.write(f"早停阈值: {delta}\n")
        f.write(f"训练集图片数量: {total_train_imgs}\n")
        f.write(f"验证集图片数量: {total_val_imgs}\n")
        f.write(f"批次大小: {batch_size}\n")
    print(f"\n=== 训练总结 ===")
    print(f"总训练轮数: {epochs}")
    print(f"实际训练轮数: {epoch + 1}")
    print(f"最佳模型轮数: {best_epoch}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    print(f"模型状态: 已保存验证损失最小的模型 (轮次 {best_epoch})")
    print(f"模型文件: {os.path.join(result_dir, 'resnet_model.pth')}")

    print(f"所有结果已保存至: {result_dir}")

    return model, history

# 运行训练，启用早停功能
model,history = resnet_train(train_loader=train_loader, val_loader=val_loader,
                             in_channels=3, out_channels=1, learning_rate=0.01,
                             epochs=200, patience=10,device='cuda')
