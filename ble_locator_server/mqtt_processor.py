from __future__ import annotations

from datetime import datetime
import logging
import threading
from typing import List, Optional

import paho.mqtt.client as mqtt

from .config_manager import ConfigManager
from .calculator import BeaconLocationCalculator
from .beacon_store import BeaconStore
from .models import BeaconReading, BluetoothRecord, LocationData
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
    def calculate_location(self, record: BluetoothRecord) -> Optional[LocationData]:
        try:
            # 将 BluetoothRecord 转为 BeaconReading 列表
            readings: List[BeaconReading] = [
                BeaconReading(mac=mac, rssi=int(rssi)) for mac, rssi in zip(record.macs, record.rssis)
            ]

            # 计算位置（LocationResult）
            lr = self.location_calculator.calculate_terminal_location(readings)
            if lr is None:
                logger.warning("位置计算失败: 计算返回None")
                return None

            # 标记时间戳，进入滤波
            lr.timestamp = lr.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lr_filtered = self.data_processing_service.filter(lr)
            if lr_filtered is None:
                logger.warning("位置计算失败: 滤波返回None")
                return None

            if lr_filtered.status in ["success", "single_beacon", "fallback"] and \
               lr_filtered.latitude is not None and lr_filtered.longitude is not None:
                location_data = LocationData(
                    device_id=record.device_id,
                    longitude=float(lr_filtered.longitude),
                    latitude=float(lr_filtered.latitude),
                    accuracy=lr_filtered.accuracy,
                    beacon_count=lr_filtered.beacon_count,
                    timestamp=record.timestamp or lr_filtered.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    calculation_method=lr_filtered.method,
                )
                logger.info(
                    "位置计算成功: (%.6f, %.6f), 方法: %s, 信标数: %s",
                    location_data.latitude,
                    location_data.longitude,
                    location_data.calculation_method,
                    location_data.beacon_count,
                )
                return location_data
            else:
                logger.warning("位置计算失败: %s", lr_filtered.message or "状态不成功或坐标缺失")
                return None
        except Exception as e:
            logger.exception("位置计算出错: %s", e)
            return None

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
                record = BluetoothRecord.parse(processed_payload)
                if record is None or not record.macs:
                    logger.warning("消息解析无有效信标数据: %s", processed_payload)
                    return
                _ = self.calculate_location(record)
        except Exception as e:
            logger.exception("处理消息时出错: %s", e)
            
