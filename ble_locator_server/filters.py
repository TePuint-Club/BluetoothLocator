from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .models import LocationResult, Position


class Filter:
    """基于 dataclass(LocationResult) 的滤波处理器"""

    def __init__(self):
        self.position_history: List[Position] = []
        self.max_history_size = 3

        # EMA滤波参数
        self.ema_alpha = 0.3
        self.ema_last_position: Optional[Position] = None

        # 卡尔曼滤波参数
        self.kf_state = None  # [lat, lon, lat_velocity, lon_velocity]
        self.kf_covariance = np.eye(4) * 0.1  # 初始协方差
        self.process_noise = 0.01  # 过程噪声
        self.measurement_noise = 0.1  # 测量噪声

    def filter(self, location_result: LocationResult) -> LocationResult:
        if location_result is None:
            return None
        lr = self._filter_middle(location_result)
        lr = self._filter_ema(lr)
        lr = self._filter_kf(lr)
        return lr

    def _filter_middle(self, lr: LocationResult) -> LocationResult:
        """
        二维中值（medoid）滤波：
        在 position_history（长度 <= max_history_size）中选出
        距离其它点总和最小的那一个作为输出。
        """
        if lr.status not in {"success", "fallback", "single_beacon"}:
            return lr
        if lr.position is None:
            return lr

        self.position_history.append(lr.position)
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        if len(self.position_history) < self.max_history_size:
            return lr

        def sqdist(p: Position, q: Position) -> float:
            return (p.latitude - q.latitude) ** 2 + (p.longitude - q.longitude) ** 2

        best_idx, best_score = 0, float("inf")
        for i, p in enumerate(self.position_history):
            score = sum(sqdist(p, q) for j, q in enumerate(self.position_history) if i != j)
            if score < best_score:
                best_idx, best_score = i, score

        lr.position = self.position_history[best_idx]
        return lr

    def _filter_ema(self, lr: LocationResult) -> LocationResult:
        """指数移动平均滤波"""
        if lr.status not in {"success", "fallback", "single_beacon"}:
            return lr
        if lr.position is None:
            return lr

        if self.ema_last_position is None:
            self.ema_last_position = lr.position
            return lr

        self.ema_last_position = Position(
            latitude=self.ema_alpha * lr.position.latitude
            + (1 - self.ema_alpha) * self.ema_last_position.latitude,
            longitude=self.ema_alpha * lr.position.longitude
            + (1 - self.ema_alpha) * self.ema_last_position.longitude,
        )
        lr.position = self.ema_last_position
        return lr

    def _filter_kf(self, lr: LocationResult) -> LocationResult:
        """卡尔曼滤波"""
        if lr.status not in {"success", "fallback", "single_beacon"}:
            return lr
        if lr.position is None:
            return lr

        if not self.kf_state:
            self.kf_state = np.array([lr.position.latitude, lr.position.longitude, 0.0, 0.0])
            return lr

        dt = 1.0  # 假设时间间隔为1秒
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.eye(4) * self.process_noise  # 过程噪声协方差矩阵
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 观测矩阵 (只观测位置，不观测速度)
        R = np.eye(2) * self.measurement_noise  # 测量噪声协方差矩阵

        # 预测步骤
        predicted_state = F @ self.kf_state
        predicted_covariance = F @ self.kf_covariance @ F.T + Q

        # 更新步骤
        measurement = np.array([lr.position.latitude, lr.position.longitude])
        innovation = measurement - H @ predicted_state
        innovation_covariance = H @ predicted_covariance @ H.T + R

        # 卡尔曼增益
        try:
            kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            kalman_gain = predicted_covariance @ H.T @ np.linalg.pinv(innovation_covariance)

        # 更新状态和协方差
        self.kf_state = predicted_state + kalman_gain @ innovation
        self.kf_covariance = (np.eye(4) - kalman_gain @ H) @ predicted_covariance

        lr.latitude = float(self.kf_state[0])
        lr.longitude = float(self.kf_state[1])
        return lr
