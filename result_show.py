import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(train_loss_file=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\train_loss.npy",
                     val_loss_file=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\val_loss.npy",
                     save_path=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\loss.jpg"):
    """
    绘制训练和验证损失曲线（只绘制到最佳验证轮次）

    参数:
    - train_loss_file: 训练损失文件路径
    - val_loss_file: 验证损失文件路径
    - save_path: 图片保存路径（可选）
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    train_loss = np.load(train_loss_file)
    val_loss = np.load(val_loss_file)

    # 确保训练和验证数据长度一致（每轮都有验证）
    min_len = min(len(train_loss), len(val_loss))
    train_loss = train_loss[:min_len]
    val_loss = val_loss[:min_len]

    # 获取最佳验证轮次
    best_epoch = np.argmin(val_loss) + 1  # +1 是因为索引从 0 开始

    # 创建图像
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # 训练和验证轮次（只到最佳轮次）
    epochs = range(1, best_epoch + 1)
    train_loss_to_best = train_loss[:best_epoch]
    val_loss_to_best = val_loss[:best_epoch]

    # 绘制训练损失曲线（只到最佳轮次）
    ax1.plot(epochs, train_loss_to_best, 'b-', label='训练损失', linewidth=2, alpha=0.8)

    # 绘制验证损失曲线（只到最佳轮次）
    ax1.plot(epochs, val_loss_to_best, 'ro-', label='验证损失', linewidth=2,
             markersize=6, markerfacecolor='red', markeredgecolor='darkred')

    # 在最佳轮次处添加标记
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'最佳验证轮次 ({best_epoch})')

    # 找到并显示训练损失的最小值
    train_min_idx = np.argmin(train_loss_to_best)
    train_min_value = train_loss_to_best[train_min_idx]

    # 找到并显示验证损失的最小值
    val_min_idx = np.argmin(val_loss_to_best)
    val_min_value = val_loss_to_best[val_min_idx]

    ax1.set_title('训练和验证损失曲线 (只显示到最佳验证轮次)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮次 (Epochs)')
    ax1.set_ylabel('损失值 (Loss)')
    ax1.legend([f'训练损失 (最小值：{train_min_value:.6f})',
                f'验证损失 (最小值：{val_min_value:.6f})',
                f'最佳验证轮次 ({best_epoch})'],
               loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 设置 x 轴刻度
    max_epoch = best_epoch
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    ax1.set_xticks(xticks)
    ax1.set_xlim(0.5, max_epoch + 0.5)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
def plot_iou_curve(iou_file=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\iou.npy",
                   save_path=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\iou.jpg"):
    """
    绘制 IoU 曲线（只绘制到最佳验证轮次）

    参数:
    - iou_file: IoU 数据文件路径
    - save_path: 图片保存路径（可选）
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    iou = np.load(iou_file)

    # 获取最佳 IoU 轮次
    best_epoch = np.argmax(iou) + 1  # +1 是因为索引从 0 开始

    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制 IoU 曲线（只到最佳轮次）
    epochs = range(1, best_epoch + 1)
    iou_to_best = iou[:best_epoch]
    ax.plot(epochs, iou_to_best, 'g-', linewidth=2, alpha=0.8)

    # 在最佳轮次处添加标记
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'最佳 IoU 轮次 ({best_epoch})')

    # 找到并显示 IoU 的最大值
    iou_max_idx = np.argmax(iou_to_best)
    iou_max_value = iou_to_best[iou_max_idx]

    ax.set_title('IoU 曲线 (只显示到最佳 IoU 轮次)', fontsize=14, fontweight='bold')
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel('IoU')
    ax.legend([f'IoU (最大值：{iou_max_value:.6f})',
               f'最佳 IoU 轮次 ({best_epoch})'],
              loc='upper right')
    ax.grid(True, alpha=0.3)

    # 设置 x 轴刻度
    max_epoch = best_epoch
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_epoch + 0.5)

    # 设置 y 轴范围
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
def plot_miou_curve(miou_file=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\miou.npy",
                    save_path=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\miou.jpg"):
    """
    绘制 mIoU 曲线（只绘制到最佳验证轮次）

    参数:
    - miou_file: mIoU 数据文件路径
    - save_path: 图片保存路径（可选）
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    miou = np.load(miou_file)

    # 获取最佳 mIoU 轮次
    best_epoch = np.argmax(miou) + 1  # +1 是因为索引从 0 开始

    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制 mIoU 曲线（只到最佳轮次）
    epochs = range(1, best_epoch + 1)
    miou_to_best = miou[:best_epoch]
    ax.plot(epochs, miou_to_best, 'm-', linewidth=2, alpha=0.8)

    # 在最佳轮次处添加标记
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'最佳 mIoU 轮次 ({best_epoch})')

    # 找到并显示 mIoU 的最大值
    miou_max_idx = np.argmax(miou_to_best)
    miou_max_value = miou_to_best[miou_max_idx]

    ax.set_title('mIoU 曲线 (只显示到最佳 mIoU 轮次)', fontsize=14, fontweight='bold')
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel('mIoU')
    ax.legend([f'mIoU (最大值：{miou_max_value:.6f})',
               f'最佳 mIoU 轮次 ({best_epoch})'],
              loc='upper right')
    ax.grid(True, alpha=0.3)

    # 设置 x 轴刻度
    max_epoch = best_epoch
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_epoch + 0.5)

    # 设置 y 轴范围
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
def plot_dice_curve(dice_file=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\dice.npy",
                    save_path=r"C:\Users\23005\Desktop\毕设\datasets2\contrast3_ResNet50_result\dice.jpg"):
    """
    绘制 Dice 系数曲线（只绘制到最佳验证轮次）

    参数:
    - dice_file: Dice 系数数据文件路径
    - save_path: 图片保存路径（可选）
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    dice = np.load(dice_file)

    # 获取最佳 Dice 轮次
    best_epoch = np.argmax(dice) + 1  # +1 是因为索引从 0 开始

    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制 Dice 系数曲线（只到最佳轮次）
    epochs = range(1, best_epoch + 1)
    dice_to_best = dice[:best_epoch]
    ax.plot(epochs, dice_to_best, 'c-', linewidth=2, alpha=0.8)

    # 在最佳轮次处添加标记
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'最佳 Dice 轮次 ({best_epoch})')

    # 找到并显示 Dice 系数的最大值
    dice_max_idx = np.argmax(dice_to_best)
    dice_max_value = dice_to_best[dice_max_idx]

    ax.set_title('Dice 系数曲线 (只显示到最佳 Dice 轮次)', fontsize=14, fontweight='bold')
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel('Dice 系数')
    ax.legend([f'Dice 系数 (最大值：{dice_max_value:.6f})',
               f'最佳 Dice 轮次 ({best_epoch})'],
              loc='upper right')
    ax.grid(True, alpha=0.3)

    # 设置 x 轴刻度
    max_epoch = best_epoch
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max_epoch + 0.5)

    # 设置 y 轴范围
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

plot_loss_curves()
plot_iou_curve()
plot_miou_curve()
plot_dice_curve()
