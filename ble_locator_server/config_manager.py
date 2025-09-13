from __future__ import annotations

import os
import yaml


DEFAULT_CONFIG_PATH = os.environ.get(
    "BLE_LOCATOR_CONFIG",
    os.path.join(".", "config", "config.yaml"),
)


class ConfigManager:
    """配置管理类，负责读写YAML配置文件"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or DEFAULT_CONFIG_PATH
        self.default_config = {
            "mqtt": {
                "ip": "localhost",
                "port": 1883,
                "topic": "/device/blueTooth/station/+",
            },
            "rssi_model": {
                "tx_power": -59,  # 1米处的RSSI值 (dBm)
                "path_loss_exponent": 2.0,  # 路径损失指数
                "a": -2.48,
                "b": 67.81,
            },
            "optimization": {
                "use_multi_start": True,
                "num_starts": 20,
                "search_radius": 0.001,
            },
            "paths": {
                "beacon_db": os.path.join(".", "beacon", "used.json"),
                "locations_csv": os.path.join(".", "output", "terminal_locations.csv"),
                "bluetooth_csv": os.path.join(".", "other_data", "bluetooth_position_data.csv"),
                "data_path_json": os.path.join(".", "config", "bluetooth_data.json"),
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

    def set_mqtt_config(self, ip, port, topic=None):
        self.config["mqtt"]["ip"] = ip
        self.config["mqtt"]["port"] = port
        if topic is not None:
            self.config["mqtt"]["topic"] = topic
        self.save_config()

    def set_rssi_model_config(self, tx_power, path_loss_exponent, a, b):
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
