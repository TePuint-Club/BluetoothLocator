from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class Beacon:
    mac: str
    latitude: float
    longitude: float
    altitude: float = 0.0


@dataclass(frozen=True)
class BeaconReading:
    mac: str
    rssi: int


@dataclass
class LocationResult:
    status: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[float] = None
    beacon_count: int = 0
    method: str = "weighted_centroid"
    message: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # 转换为与现有流程兼容的字典（过滤掉值为None的键）
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class BluetoothRecord:
    device_id: str
    macs: List[str]
    rssis: List[int]
    rotations: List[int]
    timestamp: str

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
        return cls(device_id=device_id, macs=macs, rssis=rssis, rotations=rotations, timestamp=now_str)


@dataclass(frozen=True)
class LocationData:
    device_id: str
    longitude: float
    latitude: float
    accuracy: Optional[float]
    beacon_count: int
    timestamp: str
    calculation_method: str
