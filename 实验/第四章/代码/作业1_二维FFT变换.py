#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作业1: 二维快速傅里叶变换
实现2D FFT及逆变换，展示频谱分析和图像重构
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
    
    # 使用PIL加载图像
    img = Image.open(image_path)
    
    # 转换为灰度图
    if img.mode != 'L':
        img = img.convert('L')
    
    # 转换为numpy数组
    image = np.array(img)
    
    return image


def pad_to_power_of_2(image):
    """
    将图像填充到2的幂次尺寸
    
    Args:
        image: 输入图像
        
    Returns:
        padded_image: 填充后的图像
        original_shape: 原始图像尺寸
    """
    original_shape = image.shape
    h, w = original_shape
    
    # 计算最近的2的幂次
    new_h = 2 ** int(np.ceil(np.log2(h)))
    new_w = 2 ** int(np.ceil(np.log2(w)))
    
    # 如果已经是2的幂次，直接返回
    if new_h == h and new_w == w:
        return image, original_shape
    
    # 计算填充量
    pad_h = new_h - h
    pad_w = new_w - w
    
    # 对称填充
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 使用边缘值填充
    padded_image = np.pad(image, 
                         ((pad_top, pad_bottom), (pad_left, pad_right)), 
                         mode='edge')
    
    return padded_image, original_shape


def fft2d(image):
    """
    执行二维快速傅里叶变换
    
    Args:
        image: 输入灰度图像
        
    Returns:
        fft_result: 复数频谱（未中心化）
    """
    # 转换为float64以保证精度
    image_float = image.astype(np.float64)
    
    # 执行2D FFT
    fft_result = np.fft.fft2(image_float)
    
    return fft_result


def ifft2d(fft_result):
    """
    执行二维逆快速傅里叶变换
    
    Args:
        fft_result: 频域复数数组
        
    Returns:
        reconstructed: 重构的空间域图像
    """
    # 执行2D IFFT
    reconstructed_complex = np.fft.ifft2(fft_result)
    
    # 取实部（理论上虚部应该接近0）
    reconstructed = np.real(reconstructed_complex)
    
    # 裁剪到[0, 255]范围
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed


def fft_shift(fft_result):
    """
    将零频率分量移到频谱中心
    
    Args:
        fft_result: FFT结果
        
    Returns:
        shifted: 中心化的频谱
    """
    return np.fft.fftshift(fft_result)


def compute_magnitude_spectrum(fft_result, log_scale=True):
    """
    计算幅度谱
    
    Args:
        fft_result: FFT复数结果
        log_scale: 是否使用对数尺度
        
    Returns:
        magnitude: 幅度谱（用于可视化）
    """
    # 计算幅度
    magnitude = np.abs(fft_result)
    
    # 使用对数尺度增强显示
    if log_scale:
        magnitude = np.log(magnitude + 1)  # +1避免log(0)
    
    return magnitude


def compute_phase_spectrum(fft_result):
    """
    计算相位谱
    
    Args:
        fft_result: FFT复数结果
        
    Returns:
        phase: 相位谱（弧度）
    """
    # 计算相位
    phase = np.angle(fft_result)
    
    return phase


def visualize_fft(original, magnitude, phase, reconstructed, mse):
    """
    可视化FFT变换结果
    
    Args:
        original: 原始图像
        magnitude: 幅度谱（对数尺度）
        phase: 相位谱
        reconstructed: 重构图像
        mse: 重构误差
        
    Returns:
        fig: matplotlib图形对象
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('2D Fast Fourier Transform', fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 添加图像信息
    info_text = f"Size: {original.shape[0]}×{original.shape[1]}"
    axes[0, 0].text(0.02, 0.98, info_text, transform=axes[0, 0].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 幅度谱（对数尺度）
    im1 = axes[0, 1].imshow(magnitude, cmap='hot')
    axes[0, 1].set_title('Magnitude Spectrum (Log Scale)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 添加说明
    axes[0, 1].text(0.02, 0.98, 'Low freq at center\nHigh freq at edges', 
                   transform=axes[0, 1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 相位谱
    im2 = axes[1, 0].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Phase Spectrum', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 重构图像
    axes[1, 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Reconstructed Image (IFFT)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 添加重构精度信息
    if mse < 1.0:
        accuracy_text = f"MSE: {mse:.2e}\nGood reconstruction"
        color = 'lightgreen'
    else:
        accuracy_text = f"MSE: {mse:.2e}\nReconstruction error"
        color = 'lightyellow'
    
    axes[1, 1].text(0.02, 0.98, accuracy_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    return fig


def main():
    """主程序"""
    print("=" * 60)
    print("作业1: 二维快速傅里叶变换")
    print("=" * 60)
    
    # 图像路径
    image_path = "image.jpg"
    output_dir = "../输出结果/作业1_二维FFT变换"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载图像
        print(f"\n1. 加载图像: {image_path}")
        original = load_image(image_path)
        print(f"   原始图像尺寸: {original.shape}")
        
        # 2. 填充到2的幂次
        print("\n2. 图像预处理")
        padded, original_shape = pad_to_power_of_2(original)
        if padded.shape != original_shape:
            print(f"   填充后尺寸: {padded.shape}")
        else:
            print(f"   图像尺寸已是2的幂次，无需填充")
        
        # 3. 执行FFT变换
        print("\n3. 执行二维FFT变换")
        fft_result = fft2d(padded)
        print(f"   FFT完成，频谱尺寸: {fft_result.shape}")
        
        # 4. 中心化频谱
        print("\n4. 频谱中心化")
        fft_shifted = fft_shift(fft_result)
        
        # 5. 计算频谱
        print("\n5. 计算幅度谱和相位谱")
        magnitude = compute_magnitude_spectrum(fft_shifted, log_scale=True)
        phase = compute_phase_spectrum(fft_shifted)
        print(f"   幅度谱范围: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
        print(f"   相位谱范围: [{phase.min():.2f}, {phase.max():.2f}] 弧度")
        
        # 6. 执行逆FFT验证
        print("\n6. 执行逆FFT重构图像")
        reconstructed = ifft2d(fft_result)
        
        # 验证重构精度（使用float比较避免类型转换误差）
        mse = np.mean((padded.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
        print(f"   重构MSE: {mse:.2e}")
        if mse < 1.0:  
            print(f"   ✓ FFT变换正确（MSE < 1.0）")
        else:
            print(f"   ⚠ 重构误差较大")
        
        # 7. 可视化结果
        print("\n7. 生成可视化结果")
        fig = visualize_fft(padded, magnitude, phase, reconstructed, mse)
        
        # 8. 保存结果
        print("\n8. 保存结果")
        
        # 保存频谱图
        magnitude_img = Image.fromarray((magnitude / magnitude.max() * 255).astype(np.uint8))
        magnitude_path = os.path.join(output_dir, "频谱图.png")
        magnitude_img.save(magnitude_path)
        print(f"   频谱图已保存: {magnitude_path}")
        
        # 保存重构图像
        reconstructed_img = Image.fromarray(reconstructed)
        reconstructed_path = os.path.join(output_dir, "重构图像.png")
        reconstructed_img.save(reconstructed_path)
        print(f"   重构图像已保存: {reconstructed_path}")
        
        # 保存完整结果
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
