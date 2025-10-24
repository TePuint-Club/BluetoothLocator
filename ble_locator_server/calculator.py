from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config_manager import ConfigManager
from .beacon_store import BeaconStore
from .models import (
    Beacon,
    BeaconReading,
    BluetoothRecord,
    LocationResult,
    LocationResultMethod,
    LocationResultStatus,
    Position,
    Rssi2DistanceMethod,
)


class BeaconLocationCalculator:
    """基于RSSI的蓝牙信标定位算法"""

    def __init__(
        self,
        config_manager: ConfigManager,
        beacon_store: BeaconStore,
    ):
        self.config_manager = config_manager
        rssi_config = self.config_manager.get_rssi_model_config()

        # 1米处的RSSI值 (dBm)
        self.tx_power = rssi_config.get("tx_power", -53.97)

        # 路径损耗指数
        self.path_loss_exponent = rssi_config.get("path_loss_exponent", 2.36)

        self.a = float(rssi_config.get("a", -2.48))
        self.b = float(rssi_config.get("b", 67.81))

        # 信标存储
        self.beacon_store = beacon_store
        # 默认在构造时不强制加载，交由调用方控制（mqtt_processor 会调用 load）

    # 信标读写由 BeaconStore 负责

    def update_rssi_model_params(
        self, tx_power: float, path_loss_exponent: float, a: float, b: float
    ):
        """更新RSSI模型参数并保存到配置。"""
        self.tx_power = tx_power
        self.path_loss_exponent = path_loss_exponent
        self.a = a
        self.b = b
        self.config_manager.set_rssi_model_config(tx_power, path_loss_exponent, a, b)

    def rssi_to_distance(
        self, rssi: float, method: Rssi2DistanceMethod = Rssi2DistanceMethod.IMPROVED
    ) -> float:
        """
        基于RSSI计算距离 (单位: 米)
        method: "default" 使用路径损失模型，"improved" 使用线性拟合模型
        -281-4.51x/19.97 4.412 62.3
        """

        match method:
            case Rssi2DistanceMethod.IMPROVED:
                r = (rssi + self.b) / self.a
                return max(r, 0.1)
            case Rssi2DistanceMethod.IMPROVED_PLUS:
                r = (rssi + 74.65) / -1.68 if rssi < -78 else (rssi + 64.8) / -6.48
                return max(r, 0.1)
            case Rssi2DistanceMethod.DEFAULT:
                if rssi == 0:
                    return -1.0
                exponent = (self.tx_power - rssi) / (10.0 * self.path_loss_exponent)
                distance = math.pow(10, exponent)
                return distance
            case _:
                return -1.0

    @staticmethod
    def haversine_distance(pos1: Position, pos2: Position) -> float:
        """两点球面距离（米）。"""
        R = 6_371_000.0
        phi1 = math.radians(pos1.latitude)
        phi2 = math.radians(pos2.latitude)
        dphi = math.radians(pos2.latitude - pos1.latitude)
        dlambda = math.radians(pos2.longitude - pos1.longitude)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def weight(d: float, mu: float = 6.0, sigma: float = 3.0) -> float:
        """距离权重，弱化远距离误差影响。"""
        return 0.5 + 0.5 * math.exp(-((d - mu) ** 2) / (2 * sigma**2))

    @staticmethod
    def weighted_centroid(payloads: list[tuple[Beacon, BeaconReading]]) -> Position:
        """基于RSSI权重的质心算法，不支持海拔"""
        # 将RSSI转换为权重（RSSI越高，距离越近，权重越大）
        weights: list[float] = []
        positions: list[Position] = []
        for beacon, reading in payloads:
            weight = max(0, reading.rssi + 70)  # 假设最小RSSI为-100
            weights.append(weight)
            positions.append(beacon.position)

        if sum(weights) == 0:
            # 如果所有权重都为0，使用简单平均
            lat = sum(pos.latitude for pos in positions) / len(positions)
            lon = sum(pos.longitude for pos in positions) / len(positions)
        else:
            # 否则计算加权平均
            total_weight = sum(weights)
            lat = sum(pos.latitude * w for pos, w in zip(positions, weights)) / total_weight
            lon = sum(pos.longitude * w for pos, w in zip(positions, weights)) / total_weight
        return Position(latitude=lat, longitude=lon)

    @staticmethod
    def simple_centroid(payloads: list[tuple[Beacon, BeaconReading]]) -> Position:
        """简单几何中心算法"""
        positions: list[Position] = []
        for beacon, reading in payloads:
            positions.append(beacon.position)
        lat = sum(pos.latitude for pos in positions) / len(positions)
        lon = sum(pos.longitude for pos in positions) / len(positions)
        alt = sum(pos.altitude for pos in positions) / len(positions)
        return Position(latitude=lat, longitude=lon, altitude=alt)

    def trilateration(
        self, payloads: list[tuple[Beacon, BeaconReading]], distances: list[float]
    ) -> Optional[Position]:
        """
        线性三边定位算法（二维）
        beacon_positions: [(lat1, lon1), (lat2, lon2), (lat3, lon3), ...]
        distances: [d1, d2, d3, ...]
        返回: Position 或 None
        """
        # 从payloads提取beacon位置
        beacon_positions = [
            (beacon.position.latitude, beacon.position.longitude) for beacon, _ in payloads
        ]

        # 只取前3个信标
        positions = beacon_positions[:3]
        ds = distances[:3]
        # 构造a矩阵和b矩阵
        a = np.zeros((2, 2))
        b = np.zeros((2, 1))
        for i in range(2):
            a[i][0] = 2 * (positions[i][0] - positions[2][0])
            a[i][1] = 2 * (positions[i][1] - positions[2][1])
        for i in range(2):
            b[i][0] = (
                positions[i][0] ** 2
                - positions[2][0] ** 2
                + positions[i][1] ** 2
                - positions[2][1] ** 2
                + ds[2] ** 2
                - ds[i] ** 2
            )
        try:
            # 求解线性方程组 a * [x, y]^T = b
            result = np.linalg.solve(a, b)
            lat = result[0][0]
            lon = result[1][0]
            return Position(latitude=lat, longitude=lon)
        except:
            return None

    def calculate_terminal_location(
        self,
        record: BluetoothRecord,
        method: LocationResultMethod = LocationResultMethod.WEIGHTED_CENTROID,
    ) -> LocationResult:
        """根据蓝牙读数计算终端位置：
        - 0 个信标：返回 error
        - 1 个信标：返回该信标位置（single_beacon）
        - 2 个信标：使用加权质心
        - >=3 个信标：使用 SciPy 最小化的三边定位；失败则回退加权质心
        """
        beacons: list[Beacon | None] = self.beacon_store.get_from_record(record)

        # 过滤可参与计算的Beacon
        payloads: list[tuple[Beacon, BeaconReading]] = []
        distances: list[float] = []
        for beacon, reading in zip(beacons, record):
            if beacon:
                payloads.append((beacon, reading))
                distances.append(self.rssi_to_distance(reading.rssi))

        # 没有信标：返回空结果
        if len(payloads) == 0:
            r = LocationResult.from_bluetooth_record(record)
            r.message = "无有效信标"
            r.beacon_count = 0
            return r

        # 仅一个信标：直接返回该信标位置，accuracy 取估算距离
        if len(payloads) == 1:
            r = LocationResult.from_bluetooth_record(record)
            r.status = LocationResultStatus.FUZZY
            r.message = "非精确定位"
            r.beacon_count = 1
            r.method = LocationResultMethod.SINGLE_BEACON
            return r

        # 两个信标：采用加权质心
        if len(payloads) == 2:
            r = LocationResult.from_bluetooth_record(record)
            r.position = self.weighted_centroid(payloads)
            r.status = LocationResultStatus.FUZZY
            r.beacon_count = 2
            r.accuracy = sum(distances) / len(distances)
            r.method = LocationResultMethod.WEIGHTED_CENTROID
            return r

        # 三个及以上：SciPy 三边定位
        r = LocationResult.from_bluetooth_record(record)
        r.method = method
        match method:
            case LocationResultMethod.SIMPLE_CENTROID:
                r.position = self.simple_centroid(payloads)
            case LocationResultMethod.WEIGHTED_CENTROID:
                r.position = self.weighted_centroid(payloads)
            case LocationResultMethod.TRILATERATION:
                r.position = self.trilateration(payloads, distances)

        # 如果定位失败，则回退到加权中心定位
        if r.position is None:
            r.position = self.weighted_centroid(payloads)
            r.method = LocationResultMethod.WEIGHTED_CENTROID
            r.status = LocationResultStatus.FALLBACK
        else:
            r.status = LocationResultStatus.SUCCESS

        r.beacon_count = len(payloads)
        r.accuracy = sum(distances) / len(distances)
        return r
