#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
空间滤波相关的可复用接口，基于抽象基类方便扩展。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

PadMode = Literal["reflect", "edge", "constant"]


class SpatialFilter(ABC):
    """空间域滤波器抽象基类。"""

    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def validate_input(self, image: np.ndarray) -> None:
        if image.dtype != np.uint8:
            raise TypeError(f"{self.name}: 仅支持 uint8 灰度图像")

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """执行滤波操作。"""

    def summary(self) -> str:
        return f"{self.name} Spatial Filter"


@dataclass
class MedianFilterConfig:
    """中值滤波参数配置。"""

    kernel_size: int = 3
    pad_mode: PadMode = "reflect"

    def validate(self) -> None:
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是正奇数，例如 3、5、7")
        if self.pad_mode not in {"reflect", "edge", "constant"}:
            raise ValueError("pad_mode 只能是 reflect、edge 或 constant")


class MedianFilter(SpatialFilter):
    """中值滤波器实现。"""

    def __init__(self, config: MedianFilterConfig):
        super().__init__(name="Median")
        self.config = config
        self.config.validate()

    def apply(self, image: np.ndarray) -> np.ndarray:
        self.validate_input(image)

        k = self.config.kernel_size
        pad = k // 2
        padded = np.pad(image, pad_width=pad, mode=self.config.pad_mode)
        filtered = np.empty_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + k, j : j + k]
                filtered[i, j] = np.median(region, overwrite_input=False)

        return filtered

    def summary(self) -> str:
        return f"{self.name} Filter (kernel={self.config.kernel_size}, pad={self.config.pad_mode})"


def apply_median_filter(image: np.ndarray, config: MedianFilterConfig) -> np.ndarray:
    """向后兼容的函数式接口。"""
    return MedianFilter(config).apply(image)


__all__ = [
    "SpatialFilter",
    "MedianFilterConfig",
    "MedianFilter",
    "apply_median_filter",
]

