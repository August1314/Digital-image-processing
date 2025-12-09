#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作业1: 直方图均衡化
实现直方图均衡化算法，展示原始图像、直方图、转换函数和增强结果
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib

# 设置中文字体 - 使用更可靠的方法
matplotlib.rcParams['font.sans-serif'] = ['SimSong']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


def calculate_histogram(image):
    """
    计算图像直方图
    
    Args:
        image: 灰度图像 (H, W) numpy数组
        
    Returns:
        hist: 直方图数组 (256,)
        bins: 灰度级bins (256,)
    """
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    bins = bins[:-1].astype(int)  # 使用左边界作为bin值
    return hist, bins



def histogram_equalization(image):
    """
    执行直方图均衡化
    
    Args:
        image: 灰度图像 (H, W) numpy数组
        
    Returns:
        equalized: 均衡化后的图像
        transform_func: 转换函数 (256,) - CDF映射
    """
    # 计算直方图
    hist, _ = calculate_histogram(image)
    
    # 计算概率密度函数 (PDF)
    total_pixels = image.shape[0] * image.shape[1]
    pdf = hist / total_pixels
    
    # 计算累积分布函数 (CDF)
    cdf = np.cumsum(pdf)
    
    # 归一化CDF到[0, 255]范围，这就是转换函数
    transform_func = np.round(cdf * 255).astype(np.uint8)
    
    # 应用转换函数进行像素映射
    equalized = transform_func[image]
    
    return equalized, transform_func



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



def visualize_histogram_equalization(original, hist_orig, transform_func, 
                                     enhanced, hist_enhanced):
    """
    可视化直方图均衡化结果
    
    Args:
        original: 原始图像
        hist_orig: 原始直方图
        transform_func: 转换函数
        enhanced: 增强后的图像
        hist_enhanced: 增强后的直方图
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('直方图均衡化结果', fontsize=16, fontweight='bold')
    
    # 第一行：原始图像、原始直方图、转换函数
    # 原始图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 原始直方图
    ax2 = fig.add_subplot(gs[0, 1])
    # 使用线条而不是柱状图，更容易看清集中的分布
    ax2.plot(range(256), hist_orig, color='gray', linewidth=1.5)
    ax2.fill_between(range(256), hist_orig, alpha=0.3, color='gray')
    ax2.set_title('Original Histogram', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Gray Level')
    ax2.set_ylabel('Pixel Count')
    ax2.set_xlim([0, 255])
    # 自动调整y轴范围以显示数据
    if hist_orig.max() > 0:
        ax2.set_ylim([0, hist_orig.max() * 1.1])
    ax2.grid(True, alpha=0.3)
    
    # 添加灰度值分布范围的注释
    nonzero_indices = np.where(hist_orig > 0)[0]
    if len(nonzero_indices) > 0:
        min_gray = nonzero_indices[0]
        max_gray = nonzero_indices[-1]
        range_text = f'Range: [{min_gray}, {max_gray}]'
        ax2.text(0.98, 0.98, range_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 转换函数 (CDF)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(range(256), transform_func, 'b-', linewidth=2)
    ax3.set_title('Transform Function (CDF)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Input Gray Level')
    ax3.set_ylabel('Output Gray Level')
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, 255])
    ax3.grid(True, alpha=0.3)
    
    # 第二行：增强图像、增强直方图（占两列）
    # 增强图像
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    ax4.set_title('Equalized Image', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 增强直方图（占两列）
    ax5 = fig.add_subplot(gs[1, 1:])
    # 使用线条而不是柱状图，更容易看清分布
    ax5.plot(range(256), hist_enhanced, color='blue', linewidth=1.5)
    ax5.fill_between(range(256), hist_enhanced, alpha=0.3, color='blue')
    ax5.set_title('Equalized Histogram (More Uniform Distribution)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Gray Level')
    ax5.set_ylabel('Pixel Count')
    ax5.set_xlim([0, 255])
    # 自动调整y轴范围以显示数据
    if hist_enhanced.max() > 0:
        ax5.set_ylim([0, hist_enhanced.max() * 1.1])
    ax5.grid(True, alpha=0.3)
    
    # 添加灰度值分布范围的注释
    nonzero_indices = np.where(hist_enhanced > 0)[0]
    if len(nonzero_indices) > 0:
        min_gray = nonzero_indices[0]
        max_gray = nonzero_indices[-1]
        range_text = f'Range: [{min_gray}, {max_gray}]'
        ax5.text(0.98, 0.98, range_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 添加统计信息作为文本注释
    stats_text = f'Original: mean={np.mean(original):.1f}, std={np.std(original):.1f} | '
    stats_text += f'Equalized: mean={np.mean(enhanced):.1f}, std={np.std(enhanced):.1f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig



def main():
    """主程序"""
    print("=" * 60)
    print("作业1: 直方图均衡化")
    print("=" * 60)
    
    # 图像路径
    image_path = "../图 a.jpg"
    output_dir = "../输出结果/作业1_直方图均衡化"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载图像
        print(f"\n1. 加载图像: {image_path}")
        original = load_image(image_path)
        print(f"   图像尺寸: {original.shape}")
        
        # 2. 计算原始直方图
        print("\n2. 计算原始图像直方图")
        hist_orig, bins = calculate_histogram(original)
        
        # 3. 执行直方图均衡化
        print("\n3. 执行直方图均衡化")
        enhanced, transform_func = histogram_equalization(original)
        
        # 4. 计算增强后的直方图
        print("\n4. 计算增强后图像直方图")
        hist_enhanced, _ = calculate_histogram(enhanced)
        
        # 5. 可视化结果
        print("\n5. 生成可视化结果")
        fig = visualize_histogram_equalization(
            original, hist_orig, transform_func, 
            enhanced, hist_enhanced
        )
        
        # 6. 保存结果
        print("\n6. 保存结果")
        
        # 保存增强图像
        enhanced_img = Image.fromarray(enhanced)
        enhanced_path = os.path.join(output_dir, "增强图像.png")
        enhanced_img.save(enhanced_path)
        print(f"   增强图像已保存: {enhanced_path}")
        
        # 保存可视化结果
        result_path = os.path.join(output_dir, "完整结果.png")
        fig.savefig(result_path, dpi=150, bbox_inches='tight')
        print(f"   完整结果已保存: {result_path}")
        
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
