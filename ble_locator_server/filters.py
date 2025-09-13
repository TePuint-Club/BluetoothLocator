from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .models import LocationResult


class Filter:
    """基于 dataclass(LocationResult) 的滤波处理器"""

    def __init__(self):
        # [{'latitude':..,'longitude':..}]
        self.position_history: List[Dict[str, float]] = []
        self.max_history_size = 3
        self.ema_alpha = 0.3
        self.ema_last_position: Optional[Dict[str, float]] = None
        self.kf_initialized = False
        self.kf_state = None  # [lat, lon, lat_velocity, lon_velocity]
        self.kf_covariance = None
        self.process_noise = 0.01
        self.measurement_noise = 0.1

    def filter(self, location_result: Optional[LocationResult]) -> Optional[LocationResult]:
        if location_result is None:
            return None
        lr = self._filter_middle(location_result)
        lr = self._filter_ema(lr)
        lr = self._filter_kf(lr)
        return lr

    def _filter_middle(self, lr: LocationResult) -> LocationResult:
        if lr.status not in {"success", "fallback", "single_beacon"}:
            return lr
        lat = lr.latitude
        lon = lr.longitude
        if lat is None or lon is None:
            return lr
        current_pos = {"latitude": float(lat), "longitude": float(lon)}
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        if len(self.position_history) < self.max_history_size:
            return lr

        def sqdist(p, q):
            return (p["latitude"] - q["latitude"]) ** 2 + (p["longitude"] - q["longitude"]) ** 2

        best_idx, best_score = 0, float("inf")
        for i, p in enumerate(self.position_history):
            score = sum(sqdist(p, q)
                        for j, q in enumerate(self.position_history) if i != j)
            if score < best_score:
                best_idx, best_score = i, score
        median_pos = self.position_history[best_idx]
        lr.latitude = float(median_pos["latitude"])  # type: ignore
        lr.longitude = float(median_pos["longitude"])  # type: ignore
        return lr

    def _filter_ema(self, lr: LocationResult) -> LocationResult:
        if lr.status not in {"success", "fallback", "single_beacon"}:
            return lr
        if lr.latitude is None or lr.longitude is None:
            return lr
        current_lat = float(lr.latitude)
        current_lon = float(lr.longitude)
        if self.ema_last_position is None:
            self.ema_last_position = {
                "latitude": current_lat, "longitude": current_lon}
            return lr
        filtered_lat = self.ema_alpha * current_lat + \
            (1 - self.ema_alpha) * self.ema_last_position["latitude"]
        filtered_lon = self.ema_alpha * current_lon + \
            (1 - self.ema_alpha) * self.ema_last_position["longitude"]
        self.ema_last_position = {
            "latitude": filtered_lat, "longitude": filtered_lon}
        lr.latitude = float(filtered_lat)
        lr.longitude = float(filtered_lon)
        return lr

    def _filter_kf(self, lr: LocationResult) -> LocationResult:
        if lr.status not in {"success", "fallback", "single_beacon"}:
            return lr
        if lr.latitude is None or lr.longitude is None:
            return lr
        current_lat = float(lr.latitude)
        current_lon = float(lr.longitude)
        if not self.kf_initialized:
            self.kf_state = np.array([current_lat, current_lon, 0.0, 0.0])
            self.kf_covariance = np.eye(4) * 0.1
            self.kf_initialized = True
            return lr
        dt = 1.0
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                     [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.eye(4) * self.process_noise
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * self.measurement_noise
        predicted_state = F @ self.kf_state
        predicted_covariance = F @ self.kf_covariance @ F.T + Q
        measurement = np.array([current_lat, current_lon])
        innovation = measurement - H @ predicted_state
        innovation_covariance = H @ predicted_covariance @ H.T + R
        try:
            kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(
                innovation_covariance)
        except np.linalg.LinAlgError:
            kalman_gain = predicted_covariance @ H.T @ np.linalg.pinv(
                innovation_covariance)
        self.kf_state = predicted_state + kalman_gain @ innovation
        self.kf_covariance = (np.eye(4) - kalman_gain @
                              H) @ predicted_covariance
        lr.latitude = float(self.kf_state[0])
        lr.longitude = float(self.kf_state[1])
        return lr
