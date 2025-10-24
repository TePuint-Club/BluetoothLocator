from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Iterator
from enum import Enum


@dataclass(frozen=True)
class Position:
    latitude: float
    longitude: float
    altitude: float = 0.0


@dataclass(frozen=True)
class Beacon:
    mac: str
    latitude: float
    longitude: float
    altitude: float = 0.0

    @property
    def position(self) -> Position:
        return Position(latitude=self.latitude, longitude=self.longitude)


@dataclass(frozen=True)
class BeaconReading:
    mac: str
    rssi: int
    rotations: int = 0


class Rssi2DistanceMethod(Enum):
    DEFAULT = "default"
    IMPROVED = "improved"
    IMPROVED_PLUS = "improved+"


class LocationResultMethod(Enum):
    SINGLE_BEACON = "single_beacon"
    WEIGHTED_CENTROID = "weighted_centroid"
    SIMPLE_CENTROID = "simple_centroid"
    TRILATERATION = "trilateration"


class LocationResultStatus(Enum):
    SUCCESS = "success"
    FUZZY = "fuzzy"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass
class LocationResult:
    """
    位置计算结果
    """

    device_id: str
    status: LocationResultStatus
    message: str
    timestamp: str

    beacon_count: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: float = 0.0
    accuracy: Optional[float] = None
    method: LocationResultMethod = LocationResultMethod.WEIGHTED_CENTROID

    def to_dict(self) -> Dict[str, Any]:
        # 转换为与现有流程兼容的字典（过滤掉值为None的键）
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_bluetooth_record(cls, record: BluetoothRecord) -> "LocationResult":
        return cls(
            device_id=record.device_id,
            status=LocationResultStatus.ERROR,
            message="未计算位置",
            beacon_count=len(record),
            timestamp=record.timestamp,
        )

    @property
    def position(self) -> Optional[Position]:
        if self.latitude is not None and self.longitude is not None:
            return Position(latitude=self.latitude, longitude=self.longitude)
        return None

    @position.setter
    def position(self, pos: Position | None) -> None:
        if pos is None:
            self.latitude = None
            self.longitude = None
            self.altitude = 0.0
            return
        self.latitude = pos.latitude
        self.longitude = pos.longitude
        self.altitude = pos.altitude


@dataclass(frozen=True)
class BluetoothRecord:
    """
    蓝牙扫描记录
    """

    device_id: str
    macs: List[str]
    rssis: List[int]
    rotations: List[int]
    timestamp: str

    def __len__(self) -> int:
        return len(self.macs)

    def __getitem__(self, index: int) -> BeaconReading:
        return BeaconReading(
            mac=self.macs[index], rssi=self.rssis[index], rotations=self.rotations[index]
        )

    def __iter__(self) -> Iterator[BeaconReading]:
        for mac, rssi, rotation in zip(self.macs, self.rssis, self.rotations):
            yield BeaconReading(mac=mac, rssi=rssi, rotations=rotation)

    @property
    def is_empty(self):
        return len(self) == 0

    @classmethod
    def parse(cls, data_str: str) -> Optional["BluetoothRecord"]:
        from datetime import datetime

        parts = data_str.split(";")
        if not parts or len(parts) < 2:
            return None
        device_id = parts[-1]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        macs: List[str] = []
        rssis: List[int] = []
        rotations: List[int] = []
        for item in parts[:-1]:
            fields = item.split(",")
            if len(fields) != 3:
                continue
            mac, rssi_str, rotation_str = fields
            mac = mac.lstrip("0")
            try:
                rssi = int(rssi_str)
                rotation = int(rotation_str)
            except ValueError:
                continue
            macs.append(mac)
            rssis.append(rssi)
            rotations.append(rotation)
        return cls(
            device_id=device_id, macs=macs, rssis=rssis, rotations=rotations, timestamp=now_str
        )


@dataclass(frozen=True)
class PositionProtocolData:
    """
    泛源定位协议数据结构
    Topic: BD_FANYUAN_POSITION_TOPIC
    格式：设备ID,经度,纬度,高度,预留,预留,预留,楼层,方向,步数,距离,状态,报警类型,电量,卫星解的类型,信号质量
    """

    device_id: str  # 设备ID, %4d, 1-9999
    longitude: float  # 经度, %14.10f, WGS84坐标系
    latitude: float  # 纬度, %14.10f, WGS84坐标系
    altitude: float = 0.0  # 高度, %8.2f, 米
    reserved1: float = 0.0  # 预留, %14.2f
    reserved2: float = 0.0  # 预留, %14.2f
    reserved3: float = 0.0  # 预留, %8.2f
    floor: str = "1"  # 楼层, （-2，-1，1，2，2A，3）
    direction: float = 0.0  # 方向, %8.2f, 度，以北为0度，取值范围0~360
    steps: int = 0  # 步数, 行走步数
    distance: int = 0  # 距离, 行走距离
    status: int = 0  # 状态, 0、静止；1、行走；2、跑步；3、电梯；4、扶梯；5、楼梯；6、SOS；7、自定义
    alarm_type: int = (
        0  # 报警类型, 1、聚集；2、越界；3、摔倒；4坠楼；5、超速、6、长时间静止报警；7、一键报警；8、自定义
    )
    battery: int = 100  # 电量, 0~100
    satellite_type: int = 1  # 卫星解的类型, 0=未定位，1=单点定位，2=伪距/SBAS，4固定解，5浮点解
    signal_quality: float = 1.0  # 信号质量

    @classmethod
    def from_location_result(cls, location_data: LocationResult) -> "PositionProtocolData":
        """从LocationData转换为PositionProtocolData"""
        return cls(
            device_id=location_data.device_id,
            longitude=location_data.longitude or 0.0,
            latitude=location_data.latitude or 0.0,
            altitude=location_data.altitude or 0.0,
            satellite_type=1 if location_data.accuracy else 0,  # 有精度信息则认为已定位
            signal_quality=1.0 / (location_data.accuracy + 1) if location_data.accuracy else 1.0,
        )

    def to_fanyuan_protocol_string(self) -> str:
        """转换为协议格式字符串"""
        return (
            f"{self.device_id},"
            f"{self.longitude:.10f},"
            f"{self.latitude:.10f},"
            f"{self.altitude:.2f},"
            f"{self.reserved1:.2f},"
            f"{self.reserved2:.2f},"
            f"{self.reserved3:.2f},"
            f"{self.floor},"
            f"{self.direction:.2f},"
            f"{self.steps},"
            f"{self.distance},"
            f"{self.status},"
            f"{self.alarm_type},"
            f"{self.battery},"
            f"{self.satellite_type},"
            f"{self.signal_quality}"
        )
        
    def to_beidou_protocol_string(self) -> str:
        """转换为808 MQTT协议规定的字符串"""
        return (
            f"{self.device_id},"
            f"{self.longitude:.10f},"
            f"{self.latitude:.10f},"
            f"{self.altitude:.2f},"
        )
