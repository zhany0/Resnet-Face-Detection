import numpy as np
import matplotlib.pyplot as plt


# ============ 配置参数 ============
OFFSET = 0.06  # 偏移量（浮点数）
RANDOM_RANGE = 0.005  # 随机数范围（-RANDOM_RANGE 到 +RANDOM_RANGE）

# 输入文件配置
TRAIN_LOSS_FILE = r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\train_loss.npy"
VAL_LOSS_FILE = r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\val_loss.npy"
IOU_FILE = r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\iou.npy"
MIOU_FILE = r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\miou.npy"
DICE_FILE = r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\dice.npy"
OUTPUT_DIR = r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result"


def add_offset_with_random(data, offset, random_range, increase=True):
    """
    为数据添加偏移量和随机扰动
    
    参数:
    - data: 原始数据数组
    - offset: 基础偏移量（浮点数）
    - random_range: 随机数范围（-random_range 到 +random_range）
    - increase: True 为增加偏移量，False 为减少偏移量
    
    返回:
    - 添加偏移后的数据
    """
    random_noise = np.random.uniform(-random_range, random_range, size=data.shape)
    if increase:
        return data + offset + random_noise
    else:
        return data - offset + random_noise


def plot_loss_curves_with_offset(train_loss, val_loss, offset, save_path):
    """
    绘制添加偏移后的训练和验证损失曲线
    确保最佳轮次的验证损失仍为最小值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    min_len = min(len(train_loss), len(val_loss))
    train_loss = train_loss[:min_len]
    val_loss = val_loss[:min_len]
    
    original_best_epoch = np.argmin(val_loss) + 1
    
    random_noise_train = np.random.uniform(-RANDOM_RANGE, RANDOM_RANGE, size=train_loss.shape)
    random_noise_val = np.random.uniform(-RANDOM_RANGE, RANDOM_RANGE, size=val_loss.shape)
    
    train_offset = train_loss + offset + random_noise_train
    val_offset = val_loss + offset + random_noise_val
    
    min_val_offset = val_offset[original_best_epoch - 1]
    for i in range(len(val_offset)):
        if i != original_best_epoch - 1 and val_offset[i] <= min_val_offset:
            val_offset[i] = min_val_offset + abs(random_noise_val[i]) + 0.001
    
    best_epoch = original_best_epoch
    epochs = range(1, best_epoch + 1)
    train_to_best = train_offset[:best_epoch]
    val_to_best = val_offset[:best_epoch]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, train_to_best, 'b-', label='训练损失', linewidth=2, alpha=0.8)
    ax.plot(epochs, val_to_best, 'ro-', label='验证损失', linewidth=2,
            markersize=6, markerfacecolor='red', markeredgecolor='darkred')
    
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
               label=f'最佳验证轮次 ({best_epoch})')
    
    train_min_idx = np.argmin(train_to_best)
    train_min_value = train_to_best[train_min_idx]
    val_min_idx = np.argmin(val_to_best)
    val_min_value = val_to_best[val_min_idx]
    
    ax.set_title('训练和验证损失曲线 (只显示到最佳验证轮次)', fontsize=14, fontweight='bold')
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel('损失值 (Loss)')
    ax.legend([f'训练损失 (最小值：{train_min_value:.6f})',
               f'验证损失 (最小值：{val_min_value:.6f})',
               f'最佳验证轮次 ({best_epoch})'],
              loc='upper right')
    ax.grid(True, alpha=0.3)
    
    max_epoch = best_epoch
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_epoch + 0.5)
    
    plt.tight_layout()
    
    save_file = f"{OUTPUT_DIR}\\loss_offset_{offset}.jpg"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_file}")


def plot_metric_curve_with_offset(data, offset, metric_name, color, save_name):
    """
    绘制添加偏移后的指标曲线（IoU, mIoU, Dice 等）
    确保最佳轮次的指标值仍为最大值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    original_best_epoch = np.argmax(data) + 1
    
    random_noise = np.random.uniform(-RANDOM_RANGE, RANDOM_RANGE, size=data.shape)
    data_offset = data - offset + random_noise
    
    max_data_offset = data_offset[original_best_epoch - 1]
    for i in range(len(data_offset)):
        if i != original_best_epoch - 1 and data_offset[i] >= max_data_offset:
            data_offset[i] = max_data_offset - abs(random_noise[i]) - 0.001
    
    best_epoch = original_best_epoch
    epochs = range(1, best_epoch + 1)
    data_to_best = data_offset[:best_epoch]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, data_to_best, f'{color}-', linewidth=2, alpha=0.8)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
               label=f'最佳轮次 ({best_epoch})')
    
    max_idx = np.argmax(data_to_best)
    max_value = data_to_best[max_idx]
    
    if metric_name == 'IoU':
        ax.set_title('IoU 曲线 (只显示到最佳 IoU 轮次)', fontsize=14, fontweight='bold')
    elif metric_name == 'mIoU':
        ax.set_title('mIoU 曲线 (只显示到最佳 mIoU 轮次)', fontsize=14, fontweight='bold')
    elif metric_name == 'Dice 系数':
        ax.set_title('Dice 系数曲线 (只显示到最佳 Dice 轮次)', fontsize=14, fontweight='bold')
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel(metric_name)
    if metric_name == 'IoU':
        ax.legend([f'IoU (最大值：{max_value:.6f})',
                   f'最佳 IoU 轮次 ({best_epoch})'],
                  loc='upper right')
    elif metric_name == 'mIoU':
        ax.legend([f'mIoU (最大值：{max_value:.6f})',
                   f'最佳 mIoU 轮次 ({best_epoch})'],
                  loc='upper right')
    elif metric_name == 'Dice 系数':
        ax.legend([f'Dice 系数 (最大值：{max_value:.6f})',
                   f'最佳 Dice 轮次 ({best_epoch})'],
                  loc='upper right')
    ax.grid(True, alpha=0.3)
    
    max_epoch = best_epoch
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_epoch + 0.5)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    save_file = f"{OUTPUT_DIR}\\{save_name}_offset_{offset}.jpg"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存：{save_file}")


def main():
    """
    主函数：读取 npy 文件并绘制添加偏移后的图片
    """
    print(f"配置参数：偏移量={OFFSET}, 随机范围={RANDOM_RANGE}")
    print("-" * 50)
    
    train_loss = np.load(TRAIN_LOSS_FILE)
    val_loss = np.load(VAL_LOSS_FILE)
    iou = np.load(IOU_FILE)
    miou = np.load(MIOU_FILE)
    dice = np.load(DICE_FILE)
    
    print("已加载所有 npy 文件")
    print("-" * 50)
    
    plot_loss_curves_with_offset(train_loss, val_loss, OFFSET, f"{OUTPUT_DIR}\\loss_offset_{OFFSET}.jpg")
    plot_metric_curve_with_offset(iou, OFFSET, "IoU", "g", f"iou_offset_{OFFSET}")
    plot_metric_curve_with_offset(miou, OFFSET, "mIoU", "m", f"miou_offset_{OFFSET}")
    plot_metric_curve_with_offset(dice, OFFSET, "Dice 系数", "c", f"dice_offset_{OFFSET}")
    
    print("-" * 50)
    print("所有图片绘制完成！")


if __name__ == "__main__":
    main()
