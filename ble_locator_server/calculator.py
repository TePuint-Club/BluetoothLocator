from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

from .config_manager import ConfigManager
from .beacon_store import BeaconStore
from .models import BeaconReading, LocationResult


class BeaconLocationCalculator:
    """基于RSSI的蓝牙信标定位算法"""

    def __init__(self, config_manager: Optional[ConfigManager] = None, beacon_store: Optional[BeaconStore] = None):
        self.config_manager = config_manager or ConfigManager()
        rssi_config = self.config_manager.get_rssi_model_config()
        self.tx_power = rssi_config.get("tx_power", -53.97)
        self.path_loss_exponent = rssi_config.get("path_loss_exponent", 2.36)
        self.a = rssi_config.get("a", -2.48)
        self.b = rssi_config.get("b", 67.81)

        # 信标存储
        self.beacon_store = beacon_store or BeaconStore(self.config_manager)
        # 默认在构造时不强制加载，交由调用方控制（mqtt_processor 会调用 load）

    # 信标读写由 BeaconStore 负责

    # ---------- Models ----------
    def update_rssi_model_params(self, tx_power: float, path_loss_exponent: float, a: float, b: float):
        self.tx_power = tx_power
        self.path_loss_exponent = path_loss_exponent
        self.a = a
        self.b = b
        self.config_manager.set_rssi_model_config(tx_power, path_loss_exponent, a, b)

    def rssi_to_distance(self, rssi: float, method: str = "improved", tx_power=None, path_loss_exponent=None, b=None, a=None) -> float:
        if tx_power is None:
            tx_power = getattr(self, "tx_power", -53.97)
        if path_loss_exponent is None:
            path_loss_exponent = getattr(self, "path_loss_exponent", 2.36)
        if a is None:
            a = getattr(self, "a", -2.48)
        if b is None:
            b = getattr(self, "b", 67.81)

        if method == "improved":
            r = (rssi + b) / a
            return max(r, 0.1)
        if method == "improved+":
            r = (rssi + 74.65) / -1.68 if rssi < -78 else (rssi + 64.8) / -6.48
            return max(r, 0.1)
        if method == "default":
            _ = 3 * (10 ** ((-84.38 - rssi) / (10 * 2.1447)))

        if rssi == 0:
            return -1.0
        exponent = (tx_power - rssi) / (10.0 * path_loss_exponent)
        distance = math.pow(10, exponent)
        return distance

    @staticmethod
    def weighted_centroid(beacon_positions, rssi_values):
        if not beacon_positions:
            return None
        weights = []
        for rssi in rssi_values:
            weight = max(0, rssi + 70)
            weights.append(weight)
        if sum(weights) == 0:
            lat = sum(pos[0] for pos in beacon_positions) / len(beacon_positions)
            lon = sum(pos[1] for pos in beacon_positions) / len(beacon_positions)
        else:
            total_weight = sum(weights)
            lat = sum(pos[0] * w for pos, w in zip(beacon_positions, weights)) / total_weight
            lon = sum(pos[1] * w for pos, w in zip(beacon_positions, weights)) / total_weight
        return [lat, lon]

    def _normalize_readings(self, bluetooth_readings) -> List[BeaconReading]:
        """输入可为 BeaconReading 或 dict({'mac','rssi'})，统一转换为 BeaconReading 列表。"""
        readings: List[BeaconReading] = []
        for r in bluetooth_readings:
            if isinstance(r, BeaconReading):
                readings.append(r)
            elif isinstance(r, dict) and "mac" in r and "rssi" in r:
                readings.append(BeaconReading(mac=str(r["mac"]), rssi=int(r["rssi"])) )
        return readings

    def calculate_terminal_location(self, bluetooth_readings) -> Optional[LocationResult]:
        """根据蓝牙读数计算终端位置（仅使用加权质心算法）。"""
        if not bluetooth_readings:
            return None

        readings = self._normalize_readings(bluetooth_readings)
        beacon_positions: List[List[float]] = []  # [lat, lon]
        rssi_values: List[float] = []
        distances: List[float] = []  # 仅用于给出一个粗略的accuracy

        for reading in readings:
            b = self.beacon_store.get(reading.mac)
            if b is None:
                continue
            beacon_positions.append([b.latitude, b.longitude])
            rssi_values.append(reading.rssi)
            distances.append(self.rssi_to_distance(reading.rssi))

        beacon_count = len(beacon_positions)
        if beacon_count == 0:
            return LocationResult(status="error", message="没有找到已知位置的信标", beacon_count=0)

        result = self.weighted_centroid(beacon_positions, rssi_values)
        if not result:
            return LocationResult(status="error", message="加权质心计算失败", beacon_count=beacon_count)

        accuracy = (sum(distances) / len(distances)) if distances else None
        return LocationResult(
            status="success",
            latitude=float(result[0]),
            longitude=float(result[1]),
            accuracy=accuracy,
            beacon_count=beacon_count,
            method="weighted_centroid",
        )
