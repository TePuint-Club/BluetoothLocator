from __future__ import annotations

import os
from typing import Dict, Optional, cast

import pandas as pd  # 新增

from .models import Beacon
from .config_manager import ConfigManager


class BeaconStore:
    """管理信标数据的存储与访问（pandas + CSV）"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        # 使用 DataFrame 管理，索引为 mac
        self._df = pd.DataFrame(columns=["longitude", "latitude", "altitude"])
        self._df.index.name = "mac"
        self._config = config_manager or ConfigManager()
        
    # ---- Utils ----
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "mac" not in df.columns:
            raise KeyError("CSV 文件缺少 'mac' 列")
        # 确保列完整
        for col in ["longitude", "latitude", "altitude"]:
            if col not in df.columns:
                df[col] = 0.0
            # 转为数值，非法为 NaN 的填 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # 选择列、去重、设索引与类型
        df = df[["mac", "longitude", "latitude", "altitude"]]
        df = df.drop_duplicates(subset=["mac"], keep="last").set_index("mac")
        df = df.astype({"longitude": "float64", "latitude": "float64", "altitude": "float64"}, copy=False)
        df.index.name = "mac"
        df.index = df.index.astype(str)
        return df.sort_index()

    # ---- Load/Save ----
    def load(self, beacon_file_path: Optional[str] = None):
        csv_path = beacon_file_path or self._config.get_beacon_db_path()
        try:
            if not os.path.exists(csv_path):
                self._create_sample(csv_path)
                return
            df = pd.read_csv(csv_path, dtype={"mac": str})
            self._df = self._normalize_df(df)
        except Exception:
            # 出错时也生成示例，保证系统可运行
            self._create_sample(csv_path)

    def _create_sample(self, beacon_file_path: Optional[str] = None):
        csv_path = beacon_file_path or self._config.get_beacon_db_path()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df = pd.DataFrame([
            {"mac": "EXAMPLE-BEACON", "longitude": 120.0, "latitude": 31.0, "altitude": 0.0}
        ])
        # 内存结构：索引为 mac
        self._df = self._normalize_df(df)
        # 保存为 CSV（将索引写为列 mac）
        self._df.to_csv(csv_path, index=True, index_label="mac", encoding="utf-8")

    def save(self, beacon_file_path: Optional[str] = None):
        # 保存为 CSV（mac 作为列）
        csv_path = beacon_file_path or self._config.get_beacon_db_path()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        # 保存为 CSV（将索引写为列 mac）
        self._df.to_csv(csv_path, index=True, index_label="mac", encoding="utf-8")

    # ---- CRUD ----
    def add(self, beacon: Beacon):
        # 新增或覆盖
        self._df.loc[beacon.mac, ["longitude", "latitude", "altitude"]] = [
            float(beacon.longitude),
            float(beacon.latitude),
            float(beacon.altitude),
        ]
        self.save()

    def update(self, beacon: Beacon) -> bool:
        if beacon.mac in self._df.index:
            self._df.loc[beacon.mac, ["longitude", "latitude", "altitude"]] = [
                float(beacon.longitude),
                float(beacon.latitude),
                float(beacon.altitude),
            ]
            self.save()
            return True
        return False

    def delete(self, mac: str) -> bool:
        if mac in self._df.index:
            self._df = self._df.drop(index=mac)
            self.save()
            return True
        return False

    # ---- Accessors ----
    def has(self, mac: str) -> bool:
        return mac in self._df.index

    def get(self, mac: str) -> Optional[Beacon]:
        if mac not in self._df.index:
            return None
        row = cast(pd.Series, self._df.loc[mac])
        return Beacon(
            mac=str(mac),
            latitude=float(row.at["latitude"]),
            longitude=float(row.at["longitude"]),
            altitude=float(row.at["altitude"]) if "altitude" in row else 0.0,
        )

    def all(self) -> Dict[str, Beacon]:
        result: Dict[str, Beacon] = {}
        for mac_key, row in self._df.iterrows():
            row_s = cast(pd.Series, row)
            mac_str = str(mac_key)
            result[mac_str] = Beacon(
                mac=mac_str,
                latitude=float(row_s.at["latitude"]),
                longitude=float(row_s.at["longitude"]),
                altitude=float(row_s.at["altitude"]) if "altitude" in row_s else 0.0,
            )
        return result
