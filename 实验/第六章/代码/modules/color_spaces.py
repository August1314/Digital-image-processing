#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
颜色空间转换工具，支持 RGB 与 HSI 的双向转换。
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8
_TWO_PI = 2 * np.pi


def _prepare_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是 H×W×3 的 RGB 图像")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image.astype(np.float32), 0.0, 1.0)
    raise TypeError("仅支持 uint8 或浮点格式的 RGB 图像")


def rgb_to_hsi(image: np.ndarray) -> np.ndarray:
    """将 RGB 图像转换为 HSI（H∈[0,1)，S∈[0,1]，I∈[0,1]）。"""
    rgb = _prepare_rgb(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + _EPS
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    h = np.where(b <= g, theta, _TWO_PI - theta)
    h = (h % _TWO_PI) / _TWO_PI

    min_rgb = np.minimum(np.minimum(r, g), b)
    i = (r + g + b) / 3.0
    s = np.where(i <= _EPS, 0.0, 1.0 - min_rgb / (i + _EPS))

    hsi = np.stack([h, s, np.clip(i, 0.0, 1.0)], axis=-1)
    return hsi.astype(np.float32)


def _hsi_segment_mask(h: np.ndarray, start: float, end: float) -> np.ndarray:
    """生成 H 值位于 [start, end) 区间的掩码，单位为弧度。"""
    return (h >= start) & (h < end)


def hsi_to_rgb(hsi: np.ndarray) -> np.ndarray:
    """将 HSI 图像转换回 uint8 RGB。"""
    if hsi.ndim != 3 or hsi.shape[2] != 3:
        raise ValueError("输入必须是 H×W×3 的 HSI 图像")

    h = (hsi[..., 0] % 1.0) * _TWO_PI
    s = np.clip(hsi[..., 1], 0.0, 1.0)
    i = np.clip(hsi[..., 2], 0.0, 1.0)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    # 0 <= H < 120°
    mask = _hsi_segment_mask(h, 0.0, 2 * np.pi / 3)
    h0 = h.copy()
    temp = np.cos(np.pi / 3 - h0) + _EPS
    r[mask] = i[mask] * (1 + (s[mask] * np.cos(h0[mask])) / temp[mask])
    b[mask] = i[mask] * (1 - s[mask])
    g[mask] = 3 * i[mask] - (r[mask] + b[mask])

    # 120° <= H < 240°
    mask = _hsi_segment_mask(h, 2 * np.pi / 3, 4 * np.pi / 3)
    h1 = h.copy() - 2 * np.pi / 3
    temp = np.cos(np.pi / 3 - h1) + _EPS
    g[mask] = i[mask] * (1 + (s[mask] * np.cos(h1[mask])) / temp[mask])
    r[mask] = i[mask] * (1 - s[mask])
    b[mask] = 3 * i[mask] - (r[mask] + g[mask])

    # 240° <= H < 360°
    mask = _hsi_segment_mask(h, 4 * np.pi / 3, _TWO_PI + _EPS)
    h2 = h.copy() - 4 * np.pi / 3
    temp = np.cos(np.pi / 3 - h2) + _EPS
    b[mask] = i[mask] * (1 + (s[mask] * np.cos(h2[mask])) / temp[mask])
    g[mask] = i[mask] * (1 - s[mask])
    r[mask] = 3 * i[mask] - (g[mask] + b[mask])

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


__all__ = ["rgb_to_hsi", "hsi_to_rgb"]




