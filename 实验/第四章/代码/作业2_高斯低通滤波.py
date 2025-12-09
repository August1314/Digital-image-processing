#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作业2: 高斯低通滤波
在频率域实现高斯低通滤波，展示完整处理流程
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


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
    
    img = Image.open(image_path)
    
    if img.mode != 'L':
        img = img.convert('L')
    
    image = np.array(img)
    
    return image


def create_gaussian_filter(shape, cutoff_freq, center=None):
    """
    创建高斯低通滤波器
    
    Args:
        shape: 滤波器尺寸 (H, W)
        cutoff_freq: 截止频率D0（标准差）
        center: 中心位置，默认为图像中心
        
    Returns:
        filter: 高斯低通滤波器
        
    公式: H(u,v) = exp(-D²(u,v) / (2*D0²))
    """
    h, w = shape
    
    # 默认中心为图像中心
    if center is None:
        center = (h // 2, w // 2)
    
    cy, cx = center
    
    # 创建坐标网格
    y, x = np.ogrid[0:h, 0:w]
    
    # 计算到中心的距离
    D = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # 高斯低通滤波器公式
    H = np.exp(-(D ** 2) / (2 * (cutoff_freq ** 2)))
    
    return H


def apply_frequency_filter(fft_result, filter_mask):
    """
    在频率域应用滤波器
    
    Args:
        fft_result: 中心化的FFT结果
        filter_mask: 滤波器掩码
        
    Returns:
        filtered_fft: 滤波后的频谱
    """
    # 元素乘法
    filtered_fft = fft_result * filter_mask
    
    return filtered_fft


def calculate_energy_ratio(original_fft, filtered_fft):
    """
    计算滤波后能量保留率
    
    Args:
        original_fft: 原始频谱
        filtered_fft: 滤波后频谱
        
    Returns:
        ratio: 能量保留率（0-1之间）
        
    能量定义: E = Σ|F(u,v)|²
    """
    # 计算能量（功率谱的和）
    original_energy = np.sum(np.abs(original_fft) ** 2)
    filtered_energy = np.sum(np.abs(filtered_fft) ** 2)
    
    # 计算比率（添加小值避免除零）
    ratio = filtered_energy / (original_energy + 1e-10)
    
    return ratio


def find_cutoff_for_energy(image, target_energy=0.95, tolerance=0.01):
    """
    自动寻找达到目标能量保留率的截止频率
    
    Args:
        image: 输入图像
        target_energy: 目标能量保留率（默认0.95）
        tolerance: 容差范围
        
    Returns:
        cutoff_freq: 最优截止频率
        actual_ratio: 实际能量保留率
    """
    # 执行FFT
    image_float = image.astype(np.float64)
    fft_result = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft_result)
    
    print(f"   搜索目标能量保留率: {target_energy*100:.1f}%")
    
    # 先粗略搜索找到大致范围
    best_cutoff = 50
    best_ratio = 0
    
    for D0 in range(10, min(image.shape) // 2, 5):
        filter_mask = create_gaussian_filter(image.shape, D0)
        filtered_fft = apply_frequency_filter(fft_shifted, filter_mask)
        ratio = calculate_energy_ratio(fft_shifted, filtered_fft)
        
        if abs(ratio - target_energy) < abs(best_ratio - target_energy):
            best_cutoff = D0
            best_ratio = ratio
        
        # 如果已经低于目标，停止搜索
        if ratio < target_energy - tolerance:
            break
    
    # 在最佳值附近精细搜索
    for D0 in range(max(10, best_cutoff - 10), best_cutoff + 10):
        filter_mask = create_gaussian_filter(image.shape, D0)
        filtered_fft = apply_frequency_filter(fft_shifted, filter_mask)
        ratio = calculate_energy_ratio(fft_shifted, filtered_fft)
        
        if abs(ratio - target_energy) < abs(best_ratio - target_energy):
            best_cutoff = D0
            best_ratio = ratio
    
    print(f"   找到截止频率: D0 = {best_cutoff}")
    print(f"   实际能量保留率: {best_ratio*100:.2f}%")
    
    return best_cutoff, best_ratio


def visualize_filtering(original, original_spectrum, filter_mask, 
                        filtered_spectrum, filtered_image, 
                        cutoff_freq, energy_ratio):
    """
    可视化完整滤波流程（类似教材图4.35）
    
    Args:
        original: 原始图像
        original_spectrum: 原始频谱（幅度谱）
        filter_mask: 高斯滤波器
        filtered_spectrum: 滤波后频谱（幅度谱）
        filtered_image: 滤波后图像
        cutoff_freq: 截止频率
        energy_ratio: 能量保留率
        
    Returns:
        fig: matplotlib图形对象
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Gaussian Low-Pass Filtering in Frequency Domain', 
                fontsize=16, fontweight='bold')
    
    # 第一行：原始图像、原始频谱、高斯滤波器
    # 原始图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('(a) Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 原始频谱
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(original_spectrum, cmap='hot')
    ax2.set_title('(b) Spectrum (Log Scale)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 高斯滤波器
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(filter_mask, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'(c) Gaussian Filter\nD0={cutoff_freq}', 
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 第二行：滤波后频谱、滤波后图像、能量信息
    # 滤波后频谱
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(filtered_spectrum, cmap='hot')
    ax4.set_title('(d) Filtered Spectrum', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 滤波后图像
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(filtered_image, cmap='gray', vmin=0, vmax=255)
    ax5.set_title('(e) Filtered Image', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 添加平滑效果说明
    ax5.text(0.02, 0.98, 'Smoothed\nEdges blurred', 
            transform=ax5.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 能量信息和参数
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # 显示参数信息
    info_text = f"""
Filtering Parameters:
━━━━━━━━━━━━━━━━━━━━
Filter Type: Gaussian Low-Pass
Cutoff Frequency (D0): {cutoff_freq}
Energy Retained: {energy_ratio*100:.2f}%

Image Size: {original.shape[0]}×{original.shape[1]}

Formula:
H(u,v) = exp(-D²/(2D0²))

where D is the distance from
the center of the frequency
rectangle.
    """
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig


def main():
    """主程序"""
    print("=" * 60)
    print("作业2: 高斯低通滤波")
    print("=" * 60)
    
    # 图像路径
    image_path = "image.jpg"
    output_dir = "../输出结果/作业2_高斯低通滤波"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载图像
        print(f"\n1. 加载图像: {image_path}")
        original = load_image(image_path)
        print(f"   图像尺寸: {original.shape}")
        
        # 2. 寻找95%能量保留率的截止频率
        print("\n2. 自动寻找最优截止频率")
        cutoff_freq, energy_ratio = find_cutoff_for_energy(original, target_energy=0.95)
        
        # 3. 执行FFT变换
        print("\n3. 执行FFT变换")
        image_float = original.astype(np.float64)
        fft_result = np.fft.fft2(image_float)
        fft_shifted = np.fft.fftshift(fft_result)
        print(f"   FFT完成")
        
        # 4. 创建高斯滤波器
        print("\n4. 创建高斯低通滤波器")
        gaussian_filter = create_gaussian_filter(original.shape, cutoff_freq)
        print(f"   滤波器创建完成")
        
        # 5. 应用滤波器
        print("\n5. 在频率域应用滤波器")
        filtered_fft = apply_frequency_filter(fft_shifted, gaussian_filter)
        print(f"   滤波完成")
        
        # 6. 逆FFT重构图像
        print("\n6. 执行逆FFT重构图像")
        filtered_fft_unshifted = np.fft.ifftshift(filtered_fft)
        filtered_image = np.fft.ifft2(filtered_fft_unshifted)
        filtered_image = np.real(filtered_image)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        print(f"   重构完成")
        
        # 7. 计算频谱用于可视化
        print("\n7. 计算频谱用于可视化")
        original_magnitude = np.log(np.abs(fft_shifted) + 1)
        filtered_magnitude = np.log(np.abs(filtered_fft) + 1)
        
        # 8. 可视化结果
        print("\n8. 生成可视化结果")
        fig = visualize_filtering(original, original_magnitude, gaussian_filter,
                                 filtered_magnitude, filtered_image,
                                 cutoff_freq, energy_ratio)
        
        # 9. 保存结果
        print("\n9. 保存结果")
        
        # 保存滤波后图像
        filtered_img = Image.fromarray(filtered_image)
        filtered_path = os.path.join(output_dir, "滤波后图像.png")
        filtered_img.save(filtered_path)
        print(f"   滤波后图像已保存: {filtered_path}")
        
        # 保存高斯滤波器
        filter_img = Image.fromarray((gaussian_filter * 255).astype(np.uint8))
        filter_path = os.path.join(output_dir, "高斯滤波器.png")
        filter_img.save(filter_path)
        print(f"   高斯滤波器已保存: {filter_path}")
        
        # 保存完整结果
        result_path = os.path.join(output_dir, "完整结果.png")
        fig.savefig(result_path, dpi=150, bbox_inches='tight')
        print(f"   完整结果已保存: {result_path}")
        
        # 保存频谱对比
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original_magnitude, cmap='hot')
        axes[0].set_title('Original Spectrum')
        axes[0].axis('off')
        axes[1].imshow(filtered_magnitude, cmap='hot')
        axes[1].set_title('Filtered Spectrum')
        axes[1].axis('off')
        plt.tight_layout()
        
        spectrum_path = os.path.join(output_dir, "频谱对比.png")
        fig2.savefig(spectrum_path, dpi=150, bbox_inches='tight')
        print(f"   频谱对比已保存: {spectrum_path}")
        
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
