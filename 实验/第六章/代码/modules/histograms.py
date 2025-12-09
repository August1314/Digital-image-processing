#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直方图均衡相关工具。
"""
from __future__ import annotations

import numpy as np


def _cdf_equalization(values: np.ndarray, levels: int) -> np.ndarray:
    flat = values.flatten()
    hist, _ = np.histogram(flat, bins=levels, range=(0.0, 1.0))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] == 0:
        return values
    if np.isclose(cdf[-1], cdf[0]):
        return values

    cdf_normalized = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    indices = np.clip((flat * (levels - 1)).astype(int), 0, levels - 1)
    equalized = cdf_normalized[indices]
    return equalized.reshape(values.shape)


def equalize_uint8(channel: np.ndarray) -> np.ndarray:
    """对 uint8 单通道执行直方图均衡。"""
    if channel.dtype != np.uint8:
        raise TypeError("equalize_uint8 仅支持 uint8 输入")

    normalized = channel.astype(np.float32) / 255.0
    equalized = _cdf_equalization(normalized, levels=256)
    return np.clip(equalized * 255.0, 0, 255).round().astype(np.uint8)


def equalize_float_channel(channel: np.ndarray) -> np.ndarray:
    """对 [0,1] 浮点通道执行直方图均衡。"""
    if not np.issubdtype(channel.dtype, np.floating):
        raise TypeError("equalize_float_channel 仅支持浮点数组")
    normalized = np.clip(channel, 0.0, 1.0)
    return _cdf_equalization(normalized, levels=256).astype(np.float32)


def luminance_map(image: np.ndarray) -> np.ndarray:
    """计算 RGB 图像的亮度灰度图 (Y = 0.299R + 0.587G + 0.114B)。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("luminance_map 仅接受 H×W×3 的 RGB 图像")
    rgb = image.astype(np.float32)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luminance


__all__ = ["equalize_uint8", "equalize_float_channel", "luminance_map"]


