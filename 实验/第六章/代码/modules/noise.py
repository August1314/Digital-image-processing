#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
噪声相关的可复用接口，采用面向对象设计，便于扩展新的噪声模型。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


class NoiseModel(ABC):
    """所有噪声模型的抽象基类。"""

    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def validate_input(self, image: np.ndarray) -> None:
        if image.dtype != np.uint8:
            raise TypeError(f"{self.name}: 图像必须是 uint8 类型")

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """子类需实现如何向图像注入噪声。"""

    def summary(self) -> str:
        return f"{self.name} Noise Model"


@dataclass
class SaltAndPepperConfig:
    """椒盐噪声参数配置。"""

    pa: float = 0.1  # 椒（暗点）概率
    pb: float = 0.1  # 盐（亮点）概率
    pepper_value: int = 0
    salt_value: int = 255
    random_state: Optional[int] = None

    def validate(self) -> None:
        if not (0.0 <= self.pa <= 1.0) or not (0.0 <= self.pb <= 1.0):
            raise ValueError("pa 和 pb 必须位于 [0, 1] 区间内")
        if self.pa + self.pb > 1.0:
            raise ValueError("pa + pb 不能超过 1")
        for value_name, value in {
            "pepper_value": self.pepper_value,
            "salt_value": self.salt_value,
        }.items():
            if not (0 <= value <= 255):
                raise ValueError(f"{value_name} 必须在 0-255 之间")


class SaltAndPepperNoise(NoiseModel):
    """椒盐噪声实现。"""

    def __init__(self, config: SaltAndPepperConfig):
        super().__init__(name="Salt & Pepper")
        self.config = config
        self.config.validate()

    def apply(self, image: np.ndarray) -> np.ndarray:
        self.validate_input(image)

        rng = np.random.default_rng(self.config.random_state)
        noisy_image = image.copy()
        probability_map = rng.random(image.shape)

        noisy_image[probability_map < self.config.pa] = self.config.pepper_value
        noisy_image[probability_map > 1 - self.config.pb] = self.config.salt_value

        return noisy_image

    def summary(self) -> str:
        return (
            f"{self.name} Noise (Pa={self.config.pa:.2f}, "
            f"Pb={self.config.pb:.2f}, salt={self.config.salt_value}, "
            f"pepper={self.config.pepper_value})"
        )


def add_salt_and_pepper_noise(image: np.ndarray, config: SaltAndPepperConfig) -> np.ndarray:
    """向后兼容的函数式接口。"""
    return SaltAndPepperNoise(config).apply(image)


@dataclass
class GaussianNoiseConfig:
    """高斯噪声参数配置。"""

    mean: float = 0.0
    variance: float = 10.0
    random_state: Optional[int] = None

    def validate(self) -> None:
        if self.variance < 0:
            raise ValueError("方差必须非负")


class GaussianNoise(NoiseModel):
    """高斯噪声模型。"""

    def __init__(self, config: GaussianNoiseConfig):
        super().__init__("Gaussian")
        self.config = config
        self.config.validate()

    def apply(self, image: np.ndarray) -> np.ndarray:
        self.validate_input(image)
        rng = np.random.default_rng(self.config.random_state)
        noise = rng.normal(
            loc=self.config.mean,
            scale=np.sqrt(self.config.variance),
            size=image.shape,
        )
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def summary(self) -> str:
        return f"{self.name} Noise (mean={self.config.mean}, var={self.config.variance})"


def add_gaussian_noise(image: np.ndarray, config: GaussianNoiseConfig) -> np.ndarray:
    """向后兼容的高斯噪声接口。"""
    return GaussianNoise(config).apply(image)


__all__ = [
    "NoiseModel",
    "SaltAndPepperConfig",
    "SaltAndPepperNoise",
    "add_salt_and_pepper_noise",
    "GaussianNoiseConfig",
    "GaussianNoise",
    "add_gaussian_noise",
]

