#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作业3: 位平面分析
分解图像为8个位平面并进行可视化分析
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib

# 设置中文字体 - 使用更可靠的方法
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


def extract_bit_planes(image):
    """
    提取图像的所有8个位平面
    
    Args:
        image: 灰度图像 (H, W) numpy数组
        
    Returns:
        bit_planes: 8个位平面的列表，从bit 0到bit 7
    """
    bit_planes = []
    
    # 提取每个位平面 (bit 0 到 bit 7)
    for i in range(8):
        # 使用位运算提取第i位
        bit_plane = (image >> i) & 1
        
        # 缩放到0-255以便可视化
        bit_plane_visual = bit_plane * 255
        
        bit_planes.append(bit_plane_visual.astype(np.uint8))
    
    return bit_planes



def reconstruct_from_bit_planes(bit_planes, plane_indices):
    """
    从选定的位平面重构图像
    
    Args:
        bit_planes: 8个位平面的列表
        plane_indices: 要使用的位平面索引列表 (例如 [7, 6, 5])
        
    Returns:
        reconstructed: 重构的图像
    """
    # 初始化重构图像
    reconstructed = np.zeros_like(bit_planes[0], dtype=np.uint8)
    
    # 将选定的位平面组合
    for i in plane_indices:
        # 将位平面从0-255缩放回0-1
        bit_plane = (bit_planes[i] // 255).astype(np.uint8)
        # 将该位平面加到对应的位位置
        reconstructed = reconstructed | (bit_plane << i)
    
    return reconstructed



def load_image(image_path):
    """
    加载图像并转换为灰度图
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        image: 灰度图像numpy数组
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 使用PIL加载图像
    img = Image.open(image_path)
    
    # 转换为灰度图
    if img.mode != 'L':
        img = img.convert('L')
    
    # 转换为numpy数组
    image = np.array(img)
    
    return image



def visualize_bit_planes(original, bit_planes):
    """
    可视化所有位平面
    
    Args:
        original: 原始图像
        bit_planes: 8个位平面的列表
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('位平面分析', fontsize=16, fontweight='bold')
    
    # 第一个位置显示原始图像
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('原始图像', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 添加统计信息
    info_text = f"尺寸: {original.shape}\n"
    info_text += f"灰度范围: [{np.min(original)}, {np.max(original)}]"
    axes[0, 0].text(0.02, 0.02, info_text, transform=axes[0, 0].transAxes,
                   fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 显示8个位平面 (从bit 7到bit 0)
    positions = [
        (0, 1), (0, 2),  # bit 7, bit 6
        (1, 0), (1, 1), (1, 2),  # bit 5, bit 4, bit 3
        (2, 0), (2, 1), (2, 2)   # bit 2, bit 1, bit 0
    ]
    
    for idx, (i, j) in enumerate(positions):
        bit_num = 7 - idx  # 从bit 7开始显示
        axes[i, j].imshow(bit_planes[bit_num], cmap='gray', vmin=0, vmax=255)
        
        # 标注位序号和重要性
        if bit_num >= 6:
            importance = "高"
            color = 'lightgreen'
        elif bit_num >= 3:
            importance = "中"
            color = 'lightyellow'
        else:
            importance = "低"
            color = 'lightcoral'
        
        title = f'Bit {bit_num} (MSB)' if bit_num == 7 else \
                f'Bit {bit_num} (LSB)' if bit_num == 0 else \
                f'Bit {bit_num}'
        
        axes[i, j].set_title(title, fontsize=11)
        axes[i, j].axis('off')
        
        # 添加重要性标签
        axes[i, j].text(0.02, 0.98, f'重要性: {importance}', 
                       transform=axes[i, j].transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    return fig



def visualize_reconstruction(original, bit_planes):
    """
    可视化不同位平面组合的重构效果
    
    Args:
        original: 原始图像
        bit_planes: 8个位平面的列表
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('位平面重构实验', fontsize=16, fontweight='bold')
    
    # 定义不同的重构组合
    reconstructions = [
        ([7], "仅MSB (Bit 7)"),
        ([7, 6], "高2位 (Bit 7-6)"),
        ([7, 6, 5], "高3位 (Bit 7-5)"),
        ([7, 6, 5, 4], "高4位 (Bit 7-4)"),
        ([7, 6, 5, 4, 3], "高5位 (Bit 7-3)"),
        ([7, 6, 5, 4, 3, 2], "高6位 (Bit 7-2)"),
        ([7, 6, 5, 4, 3, 2, 1], "高7位 (Bit 7-1)"),
        (list(range(8)), "全部8位")
    ]
    
    for idx, (plane_indices, title) in enumerate(reconstructions):
        row = idx // 4
        col = idx % 4
        
        # 重构图像
        reconstructed = reconstruct_from_bit_planes(bit_planes, plane_indices)
        
        # 显示
        axes[row, col].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
        
        # 计算与原图的差异
        if len(plane_indices) == 8:
            diff_text = "完整图像"
        else:
            mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
            diff_text = f"MSE: {mse:.1f}"
        
        axes[row, col].text(0.02, 0.02, diff_text, 
                           transform=axes[row, col].transAxes,
                           fontsize=8, verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig



def main():
    """主程序"""
    print("=" * 60)
    print("作业3: 位平面分析")
    print("=" * 60)
    
    # 图像路径 (使用图 a.jpg)
    image_path = "../图 a.jpg"
    output_dir = "../输出结果/作业3_位平面分析"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "位平面"), exist_ok=True)
    
    try:
        # 1. 加载图像
        print(f"\n1. 加载图像: {image_path}")
        original = load_image(image_path)
        print(f"   图像尺寸: {original.shape}")
        
        # 2. 提取位平面
        print("\n2. 提取8个位平面 (Bit 0 到 Bit 7)")
        bit_planes = extract_bit_planes(original)
        print(f"   提取完成，共{len(bit_planes)}个位平面")
        
        # 3. 可视化位平面
        print("\n3. 生成位平面可视化")
        fig1 = visualize_bit_planes(original, bit_planes)
        
        # 4. 可视化重构实验
        print("\n4. 生成位平面重构实验")
        fig2 = visualize_reconstruction(original, bit_planes)
        
        # 5. 保存结果
        print("\n5. 保存结果")
        
        # 保存各个位平面
        for i in range(8):
            bit_plane_img = Image.fromarray(bit_planes[i])
            bit_plane_path = os.path.join(output_dir, "位平面", f"bit_{i}.png")
            bit_plane_img.save(bit_plane_path)
        print(f"   位平面图像已保存到: {os.path.join(output_dir, '位平面')}")
        
        # 保存位平面可视化
        result_path1 = os.path.join(output_dir, "位平面分析.png")
        fig1.savefig(result_path1, dpi=150, bbox_inches='tight')
        print(f"   位平面分析已保存: {result_path1}")
        
        # 保存重构实验
        result_path2 = os.path.join(output_dir, "重构实验.png")
        fig2.savefig(result_path2, dpi=150, bbox_inches='tight')
        print(f"   重构实验已保存: {result_path2}")
        
        # 保存一些重构示例
        reconstructions = {
            "仅MSB": [7],
            "高4位": [7, 6, 5, 4],
            "全部8位": list(range(8))
        }
        
        for name, indices in reconstructions.items():
            reconstructed = reconstruct_from_bit_planes(bit_planes, indices)
            recon_img = Image.fromarray(reconstructed)
            recon_path = os.path.join(output_dir, f"重构_{name}.png")
            recon_img.save(recon_path)
        print(f"   重构示例已保存")
        
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print("\n位平面分析说明:")
        print("- Bit 7 (MSB): 最高有效位，包含最重要的视觉信息")
        print("- Bit 6-4: 中高位，包含主要的图像结构")
        print("- Bit 3-1: 中低位，包含细节和纹理")
        print("- Bit 0 (LSB): 最低有效位，类似随机噪声")
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
