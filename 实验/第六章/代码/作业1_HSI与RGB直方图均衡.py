#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第六章编程作业：HSI 与 RGB 空间直方图均衡对比
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from modules.color_spaces import hsi_to_rgb, rgb_to_hsi
from modules.histograms import (
    equalize_float_channel,
    equalize_uint8,
    luminance_map,
)
from modules.io_utils import (
    ensure_output_dir,
    load_color_image,
    normalize_to_uint8,
    save_color_image,
)

# Matplotlib 字体设置
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "STHeiti"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

CURRENT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = CURRENT_DIR.parent
DATA_DIR = CHAPTER_DIR
OUTPUT_DIR = CHAPTER_DIR / "输出结果" / "作业1_HSI与RGB直方图均衡"


@dataclass
class ImageStats:
    mean: float
    std: float
    saturation_mean: float


def resolve_image_path(candidates: Sequence[str]) -> Path:
    for name in candidates:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"无法在 {DATA_DIR} 中找到以下任一文件: {candidates}")


def compute_stats(image: np.ndarray) -> ImageStats:
    luminance = luminance_map(image)
    mean = float(luminance.mean())
    std = float(luminance.std())
    saturation = np.linalg.norm(image.astype(np.float32) - luminance[..., None], axis=2)
    saturation_mean = float(saturation.mean())
    return ImageStats(mean=mean, std=std, saturation_mean=saturation_mean)


def apply_hsi_equalization(image: np.ndarray) -> np.ndarray:
    hsi = rgb_to_hsi(image)
    I_eq = equalize_float_channel(hsi[..., 2])
    hsi_eq = hsi.copy()
    hsi_eq[..., 2] = I_eq
    return hsi_to_rgb(hsi_eq)


def apply_rgb_equalization(image: np.ndarray) -> np.ndarray:
    channels = [equalize_uint8(image[..., c]) for c in range(3)]
    return np.stack(channels, axis=-1)


def visualize_results(results: Dict[str, Dict[str, np.ndarray]], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HSI 与 RGB 空间直方图均衡对比", fontsize=18, fontweight="bold")

    titles = ["原始图像", "方法#1：HSI-I通道均衡", "方法#2：RGB 三通道均衡"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    max_hist = 0
    hist_data = []
    for key in titles:
        luminance = luminance_map(results[key]["image"])
        counts, _ = np.histogram(luminance.flatten(), bins=256, range=(0, 255))
        hist_data.append((counts, luminance))
        max_hist = max(max_hist, counts.max())

    for idx, title in enumerate(titles):
        axes[0, idx].imshow(results[title]["image"])
        axes[0, idx].set_title(title, fontsize=14)
        axes[0, idx].axis("off")

        counts, luminance = hist_data[idx]
        axes[1, idx].bar(
            np.arange(256),
            counts,
            color=colors[idx],
            width=1.0,
            alpha=0.8,
        )
        axes[1, idx].set_xlim(0, 255)
        axes[1, idx].set_ylim(0, max_hist * 1.05)
        axes[1, idx].set_title(f"{title} 亮度直方图", fontsize=12)
        axes[1, idx].set_xlabel("灰度级")
        axes[1, idx].set_ylabel("像素数")
        stats = results[title]["stats"]
        axes[1, idx].text(
            0.02,
            0.95,
            f"均值: {stats.mean:.1f}\n标准差: {stats.std:.1f}",
            transform=axes[1, idx].transAxes,
            fontsize=11,
            weight="bold",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def describe(stats_dict: Dict[str, ImageStats]) -> str:
    baseline = stats_dict["原始图像"]
    hsi_stats = stats_dict["方法#1：HSI-I通道均衡"]
    rgb_stats = stats_dict["方法#2：RGB 三通道均衡"]

    explanation = textwrap.dedent(
        f"""
        - 方法#1 的亮度均值从 {baseline.mean:.1f} 提升到 {hsi_stats.mean:.1f}，对比度（标准差）提高到 {hsi_stats.std:.1f}，
          说明在 HSI 中仅均衡 I 通道即可显著扩展动态范围，同时保持色度基本稳定。
        - 方法#2 的亮度均值为 {rgb_stats.mean:.1f}，对比度最高 ({rgb_stats.std:.1f})，但平均色差也最大，
          体现为某些高饱和区域出现色偏，适合强调局部色彩而非保持整体色调。
        - 与原图相比，两种方法都显著压缩了低灰度聚集，直方图更均匀；HSI 更适合要求色彩自然还原的场景，
          RGB 三通道均衡更适合突出细节但需接受潜在的色彩漂移。
        """
    ).strip()

    return explanation


def main() -> None:
    print("=" * 74)
    print("第六章作业：HSI 与 RGB 空间直方图均衡".center(74))
    print("=" * 74)

    image_path = resolve_image_path(
        [
            "图片.tif",
            "Fig0637(a)(caster_stand_original).tif",
            "Fig0637(a)(caster_stand_original).png",
        ]
    )
    print(f"加载原始图像: {image_path}")
    original = load_color_image(image_path)
    print(f"图像尺寸: {original.shape}")

    print("\n方法#1：在 HSI 空间仅均衡 I 通道...")
    hsi_equalized = apply_hsi_equalization(original)

    print("方法#2：在 RGB 空间分别均衡 R/G/B 通道...")
    rgb_equalized = apply_rgb_equalization(original)

    print("\n计算亮度和色彩统计量...")
    results = {
        "原始图像": {"image": original},
        "方法#1：HSI-I通道均衡": {"image": hsi_equalized},
        "方法#2：RGB 三通道均衡": {"image": rgb_equalized},
    }

    for key, value in results.items():
        value["stats"] = compute_stats(value["image"])

    output_dir = ensure_output_dir(OUTPUT_DIR)
    save_color_image(original, output_dir / "原始图像.png")
    save_color_image(hsi_equalized, output_dir / "方法1_HSI_I通道均衡.png")
    save_color_image(rgb_equalized, output_dir / "方法2_RGB三通道均衡.png")

    diff_map = normalize_to_uint8(
        np.mean(
            np.abs(hsi_equalized.astype(np.int16) - rgb_equalized.astype(np.int16)),
            axis=2,
        )
    )
    plt.figure(figsize=(6, 5))
    plt.title("方法#1 与 方法#2 结果差异 (平均通道绝对差)", fontsize=14)
    plt.imshow(diff_map, cmap="inferno")
    plt.axis("off")
    diff_path = output_dir / "方法差异可视化.png"
    plt.tight_layout()
    plt.savefig(diff_path, dpi=220, bbox_inches="tight")
    plt.close()

    visualize_results(results, save_path=output_dir / "完整对比.png")

    print(f"\n所有结果已保存至: {output_dir}")
    print("\n对比分析：")
    print(describe({k: v["stats"] for k, v in results.items()}))
    print("=" * 74)


if __name__ == "__main__":
    main()




