from __future__ import annotations

import os
import yaml

from typing import Callable, Any


def _env_or_default(env_key: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
    v = os.environ.get(env_key)
    if v is not None:
        try:
            return cast(v)
        except Exception:
            return v
    return default


DEFAULT_CONFIG_PATH = _env_or_default(
    "BLE_LOCATOR_CONFIG",
    os.path.join(".", "config", "config.yaml"),
)


class ConfigManager:
    """配置管理类，负责读写YAML配置文件"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or DEFAULT_CONFIG_PATH
        self.default_config = {
            "mqtt": {
                "ip": _env_or_default("BLE_MQTT_IP", "localhost"),
                "port": _env_or_default("BLE_MQTT_PORT", 1883, int),
                "uplink_topic": _env_or_default("BLE_MQTT_UPLINK_TOPIC", "/device/location/{deviceId}"),
                "downlink_topic": _env_or_default("BLE_MQTT_DOWNLINK_TOPIC", "/device/blueTooth/station/+"),
            },
            "rssi_model": {
                "tx_power": _env_or_default("BLE_RSSI_TX_POWER", -59, float),
                "path_loss_exponent": _env_or_default("BLE_RSSI_PATH_LOSS", 2.0, float),
                "a": _env_or_default("BLE_RSSI_A", -2.48, float),
                "b": _env_or_default("BLE_RSSI_B", 67.81, float),
            },
            "optimization": {
                "use_multi_start": _env_or_default(
                    "BLE_OPT_USE_MULTI_START", True, lambda v: v.lower() in ("1", "true", "yes")
                ),
                "num_starts": _env_or_default("BLE_OPT_NUM_STARTS", 20, int),
                "search_radius": _env_or_default("BLE_OPT_SEARCH_RADIUS", 0.001, float),
            },
            "paths": {
                "beacon_db": _env_or_default(
                    "BLE_PATH_BEACON_DB", os.path.join(".", "beacon", "used.csv")
                ),
            },
        }
        self.load_config()

    def load_config(self) -> None:
        """加载配置文件，如果不存在则创建默认配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
                self._merge_default_config()
            else:
                self.config = self.default_config.copy()
                self.save_config()
        except Exception:
            # 发生异常时回退到默认配置
            self.config = self.default_config.copy()
            self.save_config()

    def _merge_default_config(self) -> None:
        def merge_dict(default, current):
            for key, value in default.items():
                if key not in current:
                    current[key] = value
                elif isinstance(value, dict) and isinstance(current[key], dict):
                    merge_dict(value, current[key])

        merge_dict(self.default_config, self.config)

    def save_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.config_file) or ".", exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception:
            # 忽略保存异常
            pass

    # ---------- Accessors ----------
    def get_mqtt_config(self):
        return self.config["mqtt"]

    def get_rssi_model_config(self):
        return self.config["rssi_model"]

    def get_paths(self):
        return self.config.get("paths", {})

    def get_beacon_db_path(self):
        return self.get_paths()["beacon_db"]

    def set_mqtt_config(self, ip, port, uplink_topic=None, downlink_topic=None):
        self.config["mqtt"]["ip"] = ip
        self.config["mqtt"]["port"] = port
        if uplink_topic is not None:
            self.config["mqtt"]["uplink_topic"] = uplink_topic
        if downlink_topic is not None:
            self.config["mqtt"]["downlink_topic"] = downlink_topic
        self.save_config()

    def set_rssi_model_config(self, tx_power: float, path_loss_exponent: float, a: float, b: float):
        self.config["rssi_model"]["tx_power"] = tx_power
        self.config["rssi_model"]["path_loss_exponent"] = path_loss_exponent
        self.config["rssi_model"]["a"] = a
        self.config["rssi_model"]["b"] = b
        self.save_config()

    def get_optimization_config(self):
        return self.config.get(
            "optimization",
            {"use_multi_start": True, "num_starts": 10, "search_radius": 0.001},
        )

    def set_optimization_config(self, use_multi_start=True, num_starts=10, search_radius=0.001):
        if "optimization" not in self.config:
            self.config["optimization"] = {}
        self.config["optimization"]["use_multi_start"] = use_multi_start
        self.config["optimization"]["num_starts"] = num_starts
        self.config["optimization"]["search_radius"] = search_radius
        self.save_config()
