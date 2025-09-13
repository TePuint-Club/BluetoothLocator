from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


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
