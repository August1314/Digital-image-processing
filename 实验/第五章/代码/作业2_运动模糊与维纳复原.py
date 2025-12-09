#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第五章编程作业 #2
要求：实现式(5.77)运动模糊（T=1，方向+45°），向模糊图像添加均值0、方差10的高斯噪声，
再依据式(5.85)的维纳滤波器恢复图像。
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from modules.frequency import (
    MotionBlurConfig,
    MotionBlurDegradation,
    WienerFilter,
    WienerFilterConfig,
)
from modules.io_utils import (
    ensure_output_dir,
    load_grayscale_image,
    log_magnitude_spectrum,
    normalize_to_uint8,
    save_grayscale_image,
)
from modules.metrics import MSE, PSNR
from modules.noise import GaussianNoise, GaussianNoiseConfig

# Matplotlib 字体配置
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "STHeiti"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

CURRENT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = CURRENT_DIR.parent
OUTPUT_DIR = CHAPTER_DIR / "输出结果" / "作业2_运动模糊与维纳复原"


def resolve_image_path(candidates: Sequence[str]) -> Path:
    """依次查找可用的图像文件，兼容多种命名。"""
    for name in candidates:
        candidate = CHAPTER_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"无法在 {CHAPTER_DIR} 找到以下任一图像: {candidates}")


def visualize_results(
    original: np.ndarray,
    blurred: np.ndarray,
    noisy: np.ndarray,
    restored: np.ndarray,
    spectrum: np.ndarray,
    diff_map: np.ndarray,
    metrics_dict: Dict[str, float],
    save_path: Path,
) -> None:
    """生成 2x3 可视化面板。"""
    titles = [
        "原始图像",
        "运动模糊 (式5.77)",
        "模糊 + 高斯噪声",
        "维纳复原结果",
        "差异 |复原-原图|",
        "退化传递函数 |H(u,v)|",
    ]
    images = [original, blurred, noisy, restored, diff_map, spectrum]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("运动模糊与维纳复原实验", fontsize=16, fontweight="bold")

    for ax, title, img in zip(axes.flatten(), titles, images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    info_text = "\n".join(
        [
            f"PSNR(模糊 vs 原图): {metrics_dict['blurred_psnr']:.2f} dB",
            f"PSNR(噪声 vs 原图): {metrics_dict['noisy_psnr']:.2f} dB",
            f"PSNR(复原 vs 原图): {metrics_dict['restored_psnr']:.2f} dB",
            f"MSE(复原 vs 原图): {metrics_dict['restored_mse']:.2f}",
        ]
    )
    axes[0, 2].text(
        0.02,
        0.98,
        info_text,
        transform=axes[0, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def describe_observations(metrics_dict: Dict[str, float]) -> str:
    """生成实验结论。"""
    return textwrap.dedent(
        f"""
        - 运动模糊后图像的 PSNR 下降到 {metrics_dict['blurred_psnr']:.2f} dB，细节沿 +45° 方向被拖尾。
        - 加入方差为 10 的高斯噪声进一步将 PSNR 降至 {metrics_dict['noisy_psnr']:.2f} dB，噪点在暗背景尤为明显。
        - 维纳滤波 (K=0.01) 将 PSNR 提升至 {metrics_dict['restored_psnr']:.2f} dB，能有效抑制噪声并重建文字边缘，
          但由于频谱零点仍有限制，残留的条纹和轻微过度增强仍存在。
        """
    ).strip()


def main() -> None:
    print("=" * 70)
    print("第五章作业 #2：运动模糊与维纳复原".center(70))
    print("=" * 70)

    # 1. 加载原始图像
    image_path = resolve_image_path(["图b.png", "图b.tif", "图b.jpg", "b.png"])
    print(f"加载原始图像: {image_path}")
    original = load_grayscale_image(image_path)
    print(f"图像尺寸: {original.shape}")

    # 2. 运动模糊
    # 调整为更明显的 +45° 运动模糊，长度约 0.08 个周期
    motion_config = MotionBlurConfig(T=1, a=0.08, b=0.08)
    motion_model = MotionBlurDegradation(motion_config)
    print(f"\n应用运动模糊: {motion_model.summary()}")
    blurred, transfer = motion_model(original)

    # 3. 添加高斯噪声
    noise_config = GaussianNoiseConfig(mean=0.0, variance=10.0, random_state=7)
    noise_model = GaussianNoise(noise_config)
    print(f"添加高斯噪声: {noise_model.summary()}")
    noisy = noise_model(blurred)

    # 4. 维纳复原
    wiener_config = WienerFilterConfig(k=0.02)
    wiener_filter = WienerFilter(wiener_config)
    print(f"\n执行维纳复原: {wiener_filter.summary()}")
    restored = wiener_filter(noisy, transfer)

    # 5. 指标与可视化
    mse_metric = MSE()
    psnr_metric = PSNR()
    metrics_dict = {
        "blurred_psnr": psnr_metric(original, blurred),
        "noisy_psnr": psnr_metric(original, noisy),
        "restored_psnr": psnr_metric(original, restored),
        "restored_mse": mse_metric(original, restored),
    }
    print(
        f"\n指标: PSNR_blur={metrics_dict['blurred_psnr']:.2f} dB, "
        f"PSNR_noisy={metrics_dict['noisy_psnr']:.2f} dB, "
        f"PSNR_restored={metrics_dict['restored_psnr']:.2f} dB"
    )

    diff_map = normalize_to_uint8(
        np.abs(restored.astype(np.int16) - original.astype(np.int16))
    )
    spectrum_img = log_magnitude_spectrum(transfer)

    output_dir = ensure_output_dir(OUTPUT_DIR)
    save_grayscale_image(blurred, output_dir / "运动模糊.png")
    save_grayscale_image(noisy, output_dir / "模糊加噪.png")
    save_grayscale_image(restored, output_dir / "维纳复原.png")
    save_grayscale_image(diff_map, output_dir / "复原差异.png")
    save_grayscale_image(spectrum_img, output_dir / "传递函数幅度.png")

    visualize_results(
        original=original,
        blurred=blurred,
        noisy=noisy,
        restored=restored,
        spectrum=spectrum_img,
        diff_map=diff_map,
        metrics_dict=metrics_dict,
        save_path=output_dir / "完整结果.png",
    )

    print(f"\n所有结果已保存至: {output_dir}")
    print("\n实验观察：")
    print(describe_observations(metrics_dict))
    print("=" * 70)


if __name__ == "__main__":
    main()

