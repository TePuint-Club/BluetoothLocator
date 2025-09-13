from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
import threading
from typing import Callable, List, Optional

import paho.mqtt.client as mqtt

from .config_manager import ConfigManager
from .calculator import BeaconLocationCalculator
from .beacon_store import BeaconStore
from .models import BeaconReading
from .filters import Filter


logger = logging.getLogger(__name__)


class MQTTDataProcessor:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        # 仅保留实时处理所需的状态
        self.lock = threading.Lock()
        self.config_manager = config_manager or ConfigManager()
        # 记录/暂停逻辑与持久化已移除
        self.current_topic = None

        # 信标与计算器
        self.beacon_store = BeaconStore(self.config_manager)
        self.beacon_store.load()
        self.location_calculator = BeaconLocationCalculator(self.config_manager, self.beacon_store)

        # 过滤器
        self.data_processing_service = Filter()

    # ---------- Core processing ----------
    def handle_bluetooth_position_data(self, data_str: str) -> list[dict]:
        data = data_str.split(";")
        results = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device_id = data[-1]
        for item in data[:-1]:
            mac, rssi, rotation = item.split(",")
            results.append(
                {
                    "device_id": device_id,
                    "mac": mac,
                    "rssi": int(rssi),
                    "rotation": int(rotation),
                    "timestamp": current_time,
                }
            )
        return results

    def calculate_location_for_visualization(self, bluetooth_results: List[dict]):
        try:
            if not bluetooth_results:
                return
            readings = [BeaconReading(mac=r["mac"], rssi=int(r["rssi"])) for r in bluetooth_results]
            lr = self.location_calculator.calculate_terminal_location(readings)
            location_result_unfiltered = lr.to_dict() if lr else None
            location_result = self.data_processing_service.filter(location_result_unfiltered)
            if location_result and location_result.get("status") in ["success", "single_beacon", "fallback"]:
                location_data = {
                    "device_id": bluetooth_results[0]["device_id"],
                    "longitude": location_result["longitude"],
                    "latitude": location_result["latitude"],
                    "accuracy": location_result["accuracy"],
                    "beacon_count": location_result["beacon_count"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calculation_method": location_result["method"],
                }
                logger.info("位置: %s", location_data)
        except Exception as e:
            logger.exception("可视化位置计算出错: %s", e)

    def calculate_location(self, bluetooth_results: List[dict]):
        try:
            readings = [BeaconReading(mac=r["mac"], rssi=int(r["rssi"])) for r in bluetooth_results]
            lr = self.location_calculator.calculate_terminal_location(readings)
            location_result_unfiltered = lr.to_dict() if lr else None
            location_result = self.data_processing_service.filter(location_result_unfiltered)
            if location_result and location_result.get("status") in ["success", "single_beacon", "fallback"]:
                location_data = {
                    "device_id": bluetooth_results[0]["device_id"],
                    "longitude": location_result["longitude"],
                    "latitude": location_result["latitude"],
                    "accuracy": location_result["accuracy"],
                    "beacon_count": location_result["beacon_count"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calculation_method": location_result["method"],
                }
                logger.info(
                    "位置计算成功: (%.6f, %.6f), 方法: %s, 信标数: %s",
                    location_result["latitude"],
                    location_result["longitude"],
                    location_result["method"],
                    location_result["beacon_count"],
                )
            else:
                logger.warning(
                    "位置计算失败: %s",
                    (location_result.get("message", "未知错误") if location_result else "计算返回None"),
                )
        except Exception as e:
            logger.exception("位置计算出错: %s", e)

    # ---------- MQTT ----------
    def start_mqtt_client(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        try:
            mqtt_config = self.config_manager.get_mqtt_config()
            self.client.connect(mqtt_config["ip"], mqtt_config["port"], 60)
            self.client.loop_forever()
        except Exception as e:
            logger.error("MQTT连接错误: %s", e)

    def stop_mqtt_client(self):
        if hasattr(self, "client") and self.client:
            try:
                self.client.disconnect()
                self.client.loop_stop()
                logger.info("MQTT连接已断开")
            except Exception as e:
                logger.error("断开MQTT连接时出错: %s", e)

    def change_mqtt_topic(self, new_topic: str) -> bool:
        if hasattr(self, "client") and self.client and self.current_topic:
            try:
                self.client.unsubscribe(self.current_topic)
                logger.info("已取消订阅主题: %s", self.current_topic)
                self.client.subscribe(new_topic)
                self.current_topic = new_topic
                logger.info("已订阅新主题: %s", new_topic)
                return True
            except Exception as e:
                logger.error("更改主题订阅时出错: %s", e)
                return False
        return False

    # ---------- MQTT handlers ----------
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("成功连接到MQTT服务器")
            mqtt_config = self.config_manager.get_mqtt_config()
            topic = mqtt_config.get("topic", "/device/blueTooth/station/+")
            client.subscribe(topic)
            self.current_topic = topic
            logger.info("已订阅主题: %s", topic)
        else:
            logger.error("连接失败，返回码: %s", rc)

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            parts = payload.split(";")
            processed_parts = []
            for part in parts[:-1]:
                items = part.split(",")
                if len(items) == 3:
                    items[0] = items[0].lstrip("0")
                    processed_parts.append(",".join(items))
                else:
                    processed_parts.append(part)
            processed_payload = ";".join(processed_parts) + ";" + parts[-1]
            logger.debug("收到消息 内容: %s", processed_payload)
            with self.lock:
                bluetooth_results = self.handle_bluetooth_position_data(payload)
                self.calculate_location_for_visualization(bluetooth_results)
                self.calculate_location(bluetooth_results)
        except Exception as e:
            logger.exception("处理消息时出错: %s", e)
            
