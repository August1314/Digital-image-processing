#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像质量评估指标的抽象接口与实现。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Metric(ABC):
    """评估指标抽象基类。"""

    name: str

    def __init__(self, name: str):
        self.name = name

    def __call__(self, reference: np.ndarray, test: np.ndarray) -> float:
        return self.compute(reference, test)

    def validate_input(self, reference: np.ndarray, test: np.ndarray) -> None:
        if reference.shape != test.shape:
            raise ValueError(f"{self.name}: 两幅图像尺寸必须一致")

    @abstractmethod
    def compute(self, reference: np.ndarray, test: np.ndarray) -> float:
        """具体指标计算。"""

    def summary(self) -> str:
        return f"{self.name} Metric"


class MSE(Metric):
    """均方误差。"""

    def __init__(self):
        super().__init__("MSE")

    def compute(self, reference: np.ndarray, test: np.ndarray) -> float:
        self.validate_input(reference, test)
        diff = reference.astype(np.float32) - test.astype(np.float32)
        return float(np.mean(diff ** 2))


@dataclass
class PSNR(Metric):
    """峰值信噪比。"""

    data_range: float = 255.0

    def __post_init__(self) -> None:
        super().__init__("PSNR")

    def compute(self, reference: np.ndarray, test: np.ndarray) -> float:
        self.validate_input(reference, test)
        mse_value = MSE().compute(reference, test)
        if mse_value == 0:
            return math.inf
        return 10 * math.log10((self.data_range ** 2) / mse_value)


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """向后兼容接口。"""
    return MSE()(reference, test)


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """向后兼容接口。"""
    return PSNR(data_range=data_range)(reference, test)


__all__ = ["Metric", "MSE", "PSNR", "mse", "psnr"]

