from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .models import Beacon
from .config_manager import ConfigManager


class BeaconStore:
    """管理信标数据的存储与访问"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self._beacons: Dict[str, Beacon] = {}
        self._config = config_manager or ConfigManager()

    # ---- Load/Save ----
    def load(self, beacon_file_path: Optional[str] = None):
        if beacon_file_path is None:
            beacon_file_path = self._config.get_paths().get(
                "beacon_db", os.path.join(".", "beacon", "used.json")
            )
        assert isinstance(beacon_file_path, str)
        try:
            if os.path.exists(beacon_file_path):
                with open(beacon_file_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._beacons = {
                    mac: Beacon(
                        mac=mac,
                        latitude=float(info.get("latitude", 0.0)),
                        longitude=float(info.get("longitude", 0.0)),
                        altitude=float(info.get("altitude", 0.0)),
                    )
                    for mac, info in raw.items()
                }
            else:
                self._create_sample(beacon_file_path)
        except Exception:
            self._create_sample(beacon_file_path)

    def _create_sample(self, beacon_file_path: str):
        sample = {
            "EXAMPLE-BEACON": {"longitude": 120.0, "latitude": 31.0, "altitude": 0.0}
        }
        os.makedirs(os.path.dirname(beacon_file_path) or ".", exist_ok=True)
        with open(beacon_file_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        self._beacons = {
            mac: Beacon(mac=mac, latitude=info["latitude"], longitude=info["longitude"], altitude=info.get(
                "altitude", 0.0))
            for mac, info in sample.items()
        }

    def save(self, beacon_file_path: Optional[str] = None):
        if beacon_file_path is None:
            beacon_file_path = self._config.get_paths().get(
                "beacon_db", os.path.join(".", "beacon", "used.json")
            )
        assert isinstance(beacon_file_path, str)
        os.makedirs(os.path.dirname(beacon_file_path) or ".", exist_ok=True)
        raw = {
            mac: {"longitude": b.longitude,
                  "latitude": b.latitude, "altitude": b.altitude}
            for mac, b in self._beacons.items()
        }
        with open(beacon_file_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    # ---- CRUD ----
    def add(self, beacon: Beacon):
        self._beacons[beacon.mac] = beacon
        self.save()

    def update(self, beacon: Beacon) -> bool:
        if beacon.mac in self._beacons:
            self._beacons[beacon.mac] = beacon
            self.save()
            return True
        return False

    def delete(self, mac: str) -> bool:
        if mac in self._beacons:
            del self._beacons[mac]
            self.save()
            return True
        return False

    # ---- Accessors ----
    def has(self, mac: str) -> bool:
        return mac in self._beacons

    def get(self, mac: str) -> Optional[Beacon]:
        return self._beacons.get(mac)

    def all(self) -> Dict[str, Beacon]:
        return dict(self._beacons)
