import json
import os
import numpy as np
import cv2
import glob
from pathlib import Path
from typing import List, Dict, Union


def coco_to_labelme_json(coco_path: Union[str, List[str]], output_dir: str = "output"):
    """
    将COCO格式文件转换为LabelMe格式（支持单个文件或文件列表）

    Args:
        coco_path: COCO文件路径或文件路径列表
        output_dir: 输出目录

    Returns:
        转换的文件总数
    """

    # 处理输入参数
    if isinstance(coco_path, str):
        coco_path = [coco_path]

    if not coco_path:
        return 0

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    total_count = 0

    for coco_file_path in coco_path:
        try:
            # 加载数据
            with open(coco_file_path, 'r',) as f:
                coco_data = json.load(f)

            # 创建映射
            image_map = {img["id"]: img for img in coco_data["images"]}
            category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

            # 分组标注
            annotations_by_image = {}
            for ann in coco_data["annotations"]:
                image_id = ann["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)

            # 转换每个图像
            count = 0
            for image_id, annotations in annotations_by_image.items():
                if image_id not in image_map:
                    continue

                img_info = image_map[image_id]

                # 创建LabelMe数据
                labelme_data = {
                    "version": "5.3.0",
                    "flags": {},
                    "shapes": [],
                    "imagePath": img_info["file_name"],
                    "imageData": None,
                    "imageHeight": img_info["height"],
                    "imageWidth": img_info["width"]
                }

                # 转换标注
                for ann in annotations:
                    if ann["segmentation"]:
                        poly = ann["segmentation"][0]
                        points = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]

                        shape = {
                            "label": category_map.get(ann["category_id"], "unknown"),
                            "points": points,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        }
                        labelme_data["shapes"].append(shape)

                # 保存文件
                output_file = os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}.json")
                with open(output_file, 'w') as f:
                    json.dump(labelme_data, f, indent=2)

                count += 1

            total_count += count
            print(f"✓ {os.path.basename(coco_file_path)} - 转换了 {count} 个文件")

        except Exception as e:
            print(f"✗ {os.path.basename(coco_file_path)} - 转换失败: {e}")

    return total_count

def batch_coco_to_labelme_json(coco_dir: Union[str, List[str]], output_dir: str = "batch_labelme_output") -> Dict[str, Dict]:
    """
    批量将多个COCO文件转换为LabelMe格式（简化版）

    Args:
        coco_dir: COCO文件路径列表或包含COCO文件的目录路径
        output_dir: 输出目录，只包含COCO JSON文件路径

    Returns:
        转换结果字典，包含每个文件的转换状态和统计信息
    """

    # 处理输入参数
    if isinstance(coco_dir, str):
        # 如果是目录路径，查找所有JSON文件
        if os.path.isdir(coco_dir):
            coco_path = [os.path.join(coco_dir, f) for f in os.listdir(coco_dir)
                          if f.endswith('.json')]
        else:
            coco_path = [coco_dir]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for coco_file in coco_path:
        try:
            # 执行转换
            converted_count = coco_to_labelme_json(coco_file, output_dir)

            results[coco_file] = {
                "status": "success",
                "converted_count": converted_count,
                "output_dir": output_dir
            }

        except Exception as e:
            results[coco_file] = {
                "status": "error",
                "error": str(e),
                "converted_count": 0
            }

    return results

def labelme_json_to_mask(json_path: str, output_path: str = None) -> np.ndarray:
    """
    将LabelMe JSON文件转换为二值掩码图像

    Args:
        json_path: LabelMe JSON文件路径
        output_path: 掩码保存路径（可选）

    Returns:
        mask: 二值掩码图像 (0-背景, 255-前景)
    """

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像尺寸
    height = data['imageHeight']
    width = data['imageWidth']

    # 创建空白掩码（全黑背景）
    mask = np.zeros((height, width), dtype=np.uint8)

    # 处理每个标注形状
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)

            # 填充多边形
            cv2.fillPoly(mask, [points], 255)

    # 保存掩码（如果指定了输出路径）
    if output_path:
        cv2.imwrite(output_path, mask)
    return mask

def batch_labelme_json_to_mask(json_dir: str, output_dir: str):
    """
    批量转换LabelMe JSON文件为掩码图像

    Args:
        json_dir: JSON文件所在目录
        output_dir: 掩码输出目录
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    print(f"找到 {len(json_files)} 个JSON文件")

    success_count = 0
    for json_file in json_files:
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            output_path = os.path.join(output_dir, f'{base_name}_mask.png')

            # 转换并保存
            labelme_json_to_mask(json_file, output_path)
            success_count += 1

        except Exception as e:
            print(f"转换失败 {json_file}: {e}")

    print(f"转换完成: {success_count}/{len(json_files)} 成功")

def background_blur_gaussian(image_path, mask_path, output_path, blur_strength):
    """
    使用高斯模糊实现背景虚化
    """
    # 读取原始图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 调整掩码大小（如果需要）
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 创建前景和背景区域
    foreground = image.copy()
    background = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

    # 将掩码转换为三通道
    mask_3ch = cv2.merge([mask, mask, mask])

    # 归一化掩码（0-1范围）
    mask_normalized = mask_3ch.astype(float) / 255.0

    # 融合前景和背景
    result = foreground * mask_normalized + background * (1 - mask_normalized)
    result = result.astype(np.uint8)

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"背景虚化完成: {output_path}")

    return result

def batch_background_blur_gaussian(images_dir, masks_folder, output_folder, blur_strength=20):
    """
    对整个文件夹的图像进行批量高斯模糊背景虚化

    参数:
    images_dir: 原始图像文件夹路径
    masks_folder: 掩码图像文件夹路径
    output_folder: 输出结果文件夹路径
    blur_strength: 模糊强度（默认20）
    """

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像
    success_count = 0
    for image_path in image_files:
        try:
            # 获取文件名（不含扩展名）
            filename = Path(image_path).stem
            image_ext = Path(image_path).suffix

            # 构建对应的掩码文件路径
            # 假设掩码文件名格式: [原文件名]_mask.png
            mask_filename = f"{filename}_mask.png"
            mask_path = os.path.join(masks_folder, mask_filename)

            # 检查掩码文件是否存在
            if not os.path.exists(mask_path):
                print(f"跳过 {filename}: 掩码文件 {mask_filename} 不存在")
                continue

            # 构建输出文件路径
            output_filename = f"{filename}_blurred{image_ext}"
            output_path = os.path.join(output_folder, output_filename)

            # 应用高斯模糊背景虚化
            result = background_blur_gaussian(image_path, mask_path, output_path, blur_strength)

            success_count += 1
            print(f"成功处理: {filename}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    print(f"\n批量处理完成！成功处理 {success_count}/{len(image_files)} 个图像")
    return success_count

json_path='./json/val'
mask_path='./mask/val'
batch_labelme_json_to_mask(json_path, mask_path)
#batch_background_blur_gaussian(init_img_path,mask_path,fin_img_path,115)
