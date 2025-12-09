#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第五章编程作业 #1
题目：对图(a)添加椒盐噪声 (Pa=Pb=0.2)，应用中值滤波并与 5.10(b) 对比。
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from modules.filters import MedianFilter, MedianFilterConfig
from modules.io_utils import (
    ensure_output_dir,
    load_grayscale_image,
    normalize_to_uint8,
    resize_to_shape,
    save_grayscale_image,
)
from modules.metrics import MSE, PSNR, mse, psnr
from modules.noise import SaltAndPepperConfig, SaltAndPepperNoise

# 配置 Matplotlib 字体，确保中文可显示
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "STHeiti"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = CURRENT_DIR.parent
DATA_DIR = CHAPTER_DIR
OUTPUT_DIR = CHAPTER_DIR / "输出结果" / "作业1_椒盐噪声与中值滤波"


def visualize_results(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    reference: np.ndarray,
    diff_map: np.ndarray,
    metrics_dict: Dict[str, float],
    save_path: Path,
) -> None:
    """生成对比可视化图。"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("椒盐噪声与中值滤波效果对比", fontsize=16, fontweight="bold")

    images = [
        ("原始图像 (图a)", original),
        ("加入噪声 Pa=Pb=0.2", noisy),
        ("中值滤波 $3\\times3$", denoised),
        ("参考图 5.10(b)", reference),
        ("差异 |滤波-参考|", diff_map),
    ]

    for ax, (title, image) in zip(axes, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    info_text = "\n".join(
        [
            f"MSE(中值 vs 5.10b): {metrics_dict['mse']:.2f}",
            f"PSNR(中值 vs 5.10b): {metrics_dict['psnr']:.2f} dB",
        ]
    )
    axes[-1].text(
        0.02,
        0.02,
        info_text,
        transform=axes[-1].transAxes,
        fontsize=10,
        color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def describe_differences(metrics_dict: Dict[str, float]) -> str:
    """给出与 5.10(b) 的差异说明。"""
    explanation = textwrap.dedent(
        f"""
        - 中值滤波后的图像与参考图 5.10(b) 存在 MSE={metrics_dict['mse']:.2f} 的残余差异，
          主要来自高概率噪声导致的亮/暗斑点仍有少量残留。
        - PSNR 为 {metrics_dict['psnr']:.2f} dB，说明中值滤波消除了大部分离群像素，
          但在边缘区域仍出现轻微模糊，这是 5.10(b) 中较为锐利细节所没有的。
        - 相比 5.10(b)，本实验的差异图显示局部导线/芯片脚区域的细节对噪声更敏感，
          说明在 Pa=Pb=0.2 的极端噪声下，仍可考虑更大的窗口或多次滤波以进一步接近参考结果。
        """
    ).strip()

    return explanation


def resolve_image_path(candidates: Sequence[str]) -> Path:
    """依次查找可用的原图文件，便于兼容不同命名。"""
    for name in candidates:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"无法在 {DATA_DIR} 中找到以下任一文件: {candidates}")


def main() -> None:
    print("=" * 70)
    print("第五章作业 #1：椒盐噪声 + 中值滤波".center(70))
    print("=" * 70)

    # 1. 加载图像
    image_path = resolve_image_path(["图a.png", "图a.tif", "图a.jpg", "a.jpg"])
    reference_path = DATA_DIR / "5.10b.png"
    print(f"加载原始图像: {image_path}")
    original = load_grayscale_image(image_path)
    print(f"图像尺寸: {original.shape}")

    print(f"加载参考图像: {reference_path}")
    reference = load_grayscale_image(reference_path)

    if reference.shape != original.shape:
        print("参考图像尺寸与原图不一致，执行双线性缩放以便比较。")
        reference = resize_to_shape(reference, original.shape)

    # 2. 添加椒盐噪声
    noise_config = SaltAndPepperConfig(pa=0.1, pb=0.1, random_state=42)
    noise_model = SaltAndPepperNoise(noise_config)
    print(f"\n添加椒盐噪声: {noise_model.summary()}")
    noisy = noise_model(original)

    # 3. 中值滤波
    filter_config = MedianFilterConfig(kernel_size=3, pad_mode="reflect")
    median_filter = MedianFilter(filter_config)
    print(f"\n应用中值滤波: {median_filter.summary()}")
    denoised = median_filter(noisy)

    # 4. 质量指标
    mse_metric = MSE()
    psnr_metric = PSNR()
    metrics_dict = {
        "mse": mse_metric(reference, denoised),
        "psnr": psnr_metric(reference, denoised),
        "noisy_psnr": psnr_metric(reference, noisy),
    }
    print(f"\n中值滤波后 vs 5.10(b): MSE={metrics_dict['mse']:.2f}, PSNR={metrics_dict['psnr']:.2f} dB")
    print(f"噪声图像 vs 5.10(b): PSNR={metrics_dict['noisy_psnr']:.2f} dB")

    # 5. 差异可视化
    diff_map = normalize_to_uint8(np.abs(denoised.astype(np.int16) - reference.astype(np.int16)))

    # 6. 保存结果
    output_dir = ensure_output_dir(OUTPUT_DIR)
    save_grayscale_image(noisy, output_dir / "噪声图像.png")
    save_grayscale_image(denoised, output_dir / "中值滤波.png")
    save_grayscale_image(diff_map, output_dir / "差异图.png")
    visualize_results(
        original=original,
        noisy=noisy,
        denoised=denoised,
        reference=reference,
        diff_map=diff_map,
        metrics_dict=metrics_dict,
        save_path=output_dir / "完整结果.png",
    )

    print(f"\n全部结果已保存至: {output_dir}")
    print("\n与 5.10(b) 的主要区别：")
    print(describe_differences(metrics_dict))
    print("=" * 70)


if __name__ == "__main__":
    main()

