#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作业2: 拉普拉斯锐化
使用拉普拉斯算子对图像进行锐化处理
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


def get_laplacian_kernel(kernel_type='standard'):
    """
    获取拉普拉斯卷积核
    
    Args:
        kernel_type: 核类型
            'standard': 作业要求指定的拉普拉斯核（包含对角线）
            
    Returns:
        kernel: 3x3拉普拉斯核
    """
    # 作业要求指定的拉普拉斯核（包含对角线的8邻域核）
    # [-1, -1, -1]
    # [-1,  8, -1]
    # [-1, -1, -1]
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    return kernel



def laplacian_filter(image, kernel):
    """
    应用拉普拉斯滤波器
    
    Args:
        image: 输入灰度图像
        kernel: 拉普拉斯卷积核
        
    Returns:
        filtered: 拉普拉斯滤波结果
    """
    # 获取图像和核的尺寸
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # 使用反射填充处理边界
    padded = np.pad(image.astype(np.float32), 
                    ((pad_h, pad_h), (pad_w, pad_w)), 
                    mode='reflect')
    
    # 初始化输出
    filtered = np.zeros_like(image, dtype=np.float32)
    
    # 执行卷积
    for i in range(h):
        for j in range(w):
            # 提取邻域
            region = padded[i:i+kh, j:j+kw]
            # 卷积计算
            filtered[i, j] = np.sum(region * kernel)
    
    return filtered



def laplacian_sharpen(image, kernel):
    """
    执行拉普拉斯锐化
    
    根据式(3.54)实现锐化：
    - 如果使用标准核(中心为正): g(x,y) = f(x,y) - ∇²f(x,y)
    - 如果使用负核(中心为负): g(x,y) = f(x,y) + ∇²f(x,y)
    
    Args:
        image: 输入灰度图像
        kernel: 拉普拉斯卷积核
        
    Returns:
        laplacian_response: 拉普拉斯响应
        sharpened: 锐化后的图像
    """
    # 应用拉普拉斯滤波
    laplacian_response = laplacian_filter(image, kernel)
    
    # 根据式(3.54)和核的类型决定锐化公式
    # 由于使用的是正核（中心为+8），锐化公式为：g(x,y) = f(x,y) + ∇²f(x,y)
    sharpened = image.astype(np.float32) + laplacian_response
    
    # 裁剪到[0, 255]范围
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # 归一化拉普拉斯响应以便可视化
    laplacian_display = laplacian_response.copy()
    laplacian_display = (laplacian_display - laplacian_display.min())
    if laplacian_display.max() > 0:
        laplacian_display = laplacian_display / laplacian_display.max() * 255
    laplacian_display = laplacian_display.astype(np.uint8)
    
    return laplacian_display, sharpened



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



def visualize_laplacian_sharpening(original, laplacian_response, sharpened, kernel):
    """
    可视化拉普拉斯锐化结果
    
    Args:
        original: 原始图像
        laplacian_response: 拉普拉斯响应
        sharpened: 锐化后的图像
        kernel: 使用的拉普拉斯核
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('拉普拉斯锐化结果', fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')
    
    # 拉普拉斯响应
    axes[1].imshow(laplacian_response, cmap='gray')
    axes[1].set_title('拉普拉斯滤波响应', fontsize=12)
    axes[1].axis('off')
    
    # 添加核的显示
    kernel_text = "使用的拉普拉斯核:\n"
    for row in kernel:
        kernel_text += "  " + "  ".join([f"{int(x):2d}" for x in row]) + "\n"
    axes[1].text(0.02, 0.98, kernel_text, transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 锐化图像
    axes[2].imshow(sharpened, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('锐化后图像', fontsize=12)
    axes[2].axis('off')
    
    # 添加统计信息
    info_text = f"对比度提升\n边缘增强"
    axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    return fig



def main():
    """主程序"""
    print("=" * 60)
    print("作业2: 拉普拉斯锐化")
    print("=" * 60)
    
    # 图像路径
    image_path = "../图 b.jpg"
    output_dir = "../输出结果/作业2_拉普拉斯锐化"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载图像
        print(f"\n1. 加载图像: {image_path}")
        original = load_image(image_path)
        print(f"   图像尺寸: {original.shape}")
        
        # 2. 获取拉普拉斯核
        print("\n2. 获取拉普拉斯卷积核（作业要求指定）")
        kernel = get_laplacian_kernel('standard')
        print("   拉普拉斯核（包含对角线的8邻域核）:")
        for row in kernel:
            print("   ", row)
        
        # 3. 执行拉普拉斯锐化
        print("\n3. 执行拉普拉斯锐化")
        laplacian_response, sharpened = laplacian_sharpen(original, kernel)
        print("   锐化完成")
        
        # 4. 可视化结果
        print("\n4. 生成可视化结果")
        fig = visualize_laplacian_sharpening(
            original, laplacian_response, sharpened, kernel
        )
        
        # 5. 保存结果
        print("\n5. 保存结果")
        
        # 保存锐化图像
        sharpened_img = Image.fromarray(sharpened)
        sharpened_path = os.path.join(output_dir, "锐化图像.png")
        sharpened_img.save(sharpened_path)
        print(f"   锐化图像已保存: {sharpened_path}")
        
        # 保存拉普拉斯响应
        laplacian_img = Image.fromarray(laplacian_response)
        laplacian_path = os.path.join(output_dir, "拉普拉斯响应.png")
        laplacian_img.save(laplacian_path)
        print(f"   拉普拉斯响应已保存: {laplacian_path}")
        
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
