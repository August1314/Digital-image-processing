#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
频率域退化与复原模型。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


class FrequencyDegradationModel(ABC):
    """频率域退化模型基类。"""

    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply(image)

    def validate_input(self, image: np.ndarray) -> None:
        if image.ndim != 2:
            raise ValueError(f"{self.name}: 仅支持灰度图像")

    @abstractmethod
    def transfer_function(self, shape: Tuple[int, int]) -> np.ndarray:
        """返回指定尺寸的退化传递函数 H(u,v)。"""

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.validate_input(image)
        H = self.transfer_function(image.shape)
        F = np.fft.fft2(image.astype(np.float32))
        G = H * F
        blurred = np.fft.ifft2(G)
        blurred = np.real(blurred)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        return blurred, H

    def summary(self) -> str:
        return f"{self.name} Degradation Model"


class FrequencyRestorationModel(ABC):
    """频率域复原模型基类。"""

    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, degraded: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        return self.restore(degraded, transfer)

    def validate_inputs(self, degraded: np.ndarray, transfer: np.ndarray) -> None:
        if degraded.shape != transfer.shape:
            raise ValueError(f"{self.name}: 图像与传递函数尺寸不匹配")

    @abstractmethod
    def restore(self, degraded: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        """执行复原。"""

    def summary(self) -> str:
        return f"{self.name} Restoration Model"


@dataclass
class MotionBlurConfig:
    """线性运动模糊配置。"""

    T: float = 1.0
    a: float = 0.1
    b: float = 0.1

    def validate(self) -> None:
        if self.T <= 0:
            raise ValueError("T 必须为正数")


class MotionBlurDegradation(FrequencyDegradationModel):
    """基于式(5.77)的线性运动模糊模型。"""

    def __init__(self, config: MotionBlurConfig):
        super().__init__("Motion Blur")
        self.config = config
        self.config.validate()

    def transfer_function(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        构造离散运动模糊传递函数。

        参考 DIP3e Eq.(5.77)，u、v 使用中心化的频率索引，随后通过 ifftshift
        对齐到 FFT 的频率排列，避免参数变化无效的问题。
        """
        M, N = shape
        u = np.arange(M) - M // 2
        v = np.arange(N) - N // 2
        U = u[:, None].astype(np.float32)
        V = v[None, :].astype(np.float32)

        pi_term = np.pi * (U * self.config.a + V * self.config.b)
        H = np.zeros_like(pi_term, dtype=np.complex64)

        nonzero = np.abs(pi_term) > 1e-8
        H[nonzero] = (
            self.config.T
            * np.sin(pi_term[nonzero])
            / (pi_term[nonzero])
            * np.exp(-1j * pi_term[nonzero])
        )
        H[~nonzero] = self.config.T

        # 将中心化频谱移动到 FFT 默认顺序
        return np.fft.ifftshift(H)

    def summary(self) -> str:
        return f"{self.name} (T={self.config.T}, a={self.config.a}, b={self.config.b})"


@dataclass
class WienerFilterConfig:
    """维纳滤波器配置。"""

    k: float = 0.01

    def validate(self) -> None:
        if self.k < 0:
            raise ValueError("k 必须非负")


class WienerFilter(FrequencyRestorationModel):
    """式(5.85) 维纳滤波实现。"""

    def __init__(self, config: WienerFilterConfig):
        super().__init__("Wiener")
        self.config = config
        self.config.validate()

    def restore(self, degraded: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        self.validate_inputs(degraded, transfer)
        G = np.fft.fft2(degraded.astype(np.float32))
        H_conj = np.conj(transfer)
        denom = (np.abs(transfer) ** 2) + self.config.k
        eps = 1e-8
        denom = np.where(denom == 0, eps, denom)
        F_hat = (H_conj / denom) * G
        restored = np.real(np.fft.ifft2(F_hat))
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        return restored

    def summary(self) -> str:
        return f"{self.name} Filter (k={self.config.k})"


__all__ = [
    "FrequencyDegradationModel",
    "FrequencyRestorationModel",
    "MotionBlurConfig",
    "MotionBlurDegradation",
    "WienerFilterConfig",
    "WienerFilter",
]

