#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像 IO 与辅助工具。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def load_grayscale_image(image_path: os.PathLike | str) -> np.ndarray:
    """加载图像并确保为 uint8 灰度格式。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    image = Image.open(path)
    if image.mode != "L":
        image = image.convert("L")

    return np.array(image, dtype=np.uint8)


def load_color_image(image_path: os.PathLike | str) -> np.ndarray:
    """加载 RGB 彩色图像，返回 uint8 格式。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return np.array(image, dtype=np.uint8)


def save_grayscale_image(image: np.ndarray, save_path: os.PathLike | str) -> None:
    """保存 uint8 灰度图像。"""
    if image.dtype != np.uint8:
        raise TypeError("仅支持保存 uint8 格式的灰度图像")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_color_image(image: np.ndarray, save_path: os.PathLike | str) -> None:
    """保存 uint8 彩色图像。"""
    if image.dtype != np.uint8:
        raise TypeError("仅支持保存 uint8 格式的彩色图像")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def ensure_output_dir(dir_path: os.PathLike | str) -> Path:
    """确保输出目录存在并返回 Path 对象。"""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """将任意浮点数组线性归一化到 [0, 255] 并转换为 uint8。"""
    arr = array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr *= 255
    return arr.astype(np.uint8)


def resize_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """将灰度图像缩放到目标尺寸。"""
    pil_image = Image.fromarray(image)
    h, w = target_shape
    resized = pil_image.resize((w, h), Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def log_magnitude_spectrum(freq_data: np.ndarray) -> np.ndarray:
    """计算频谱的对数幅度并归一化为 uint8。"""
    shifted = np.fft.fftshift(freq_data)
    magnitude = np.log1p(np.abs(shifted))
    return normalize_to_uint8(magnitude)


__all__ = [
    "load_grayscale_image",
    "load_color_image",
    "save_grayscale_image",
    "save_color_image",
    "ensure_output_dir",
    "normalize_to_uint8",
    "resize_to_shape",
    "log_magnitude_spectrum",
]

