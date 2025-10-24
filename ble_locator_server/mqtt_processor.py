from __future__ import annotations

from datetime import datetime
import logging
import threading
from typing import List, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage

from .config_manager import ConfigManager
from .calculator import BeaconLocationCalculator
from .beacon_store import BeaconStore
from .models import BluetoothRecord, LocationResult, LocationResultStatus, PositionProtocolData
from .filters import Filter


logger = logging.getLogger(__name__)


class MQTTDataProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.lock = threading.Lock()
        self.config_manager = config_manager

        # 信标与计算器
        self.beacon_store = BeaconStore(self.config_manager)
        self.beacon_store.load()
        self.location_calculator = BeaconLocationCalculator(self.config_manager, self.beacon_store)

        # 过滤器
        self.data_processing_service: dict[str, Filter] = {}

    # ---------- Core processing ----------
    def calculate_location(self, record: BluetoothRecord) -> Optional[LocationResult]:
        try:
            # 计算位置（LocationResult）
            lr = self.location_calculator.calculate_terminal_location(record)

            # 标记时间戳，进入滤波
            lr.timestamp = lr.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 为每个设备创建独立的滤波器
            device_id = record.device_id
            if device_id not in self.data_processing_service:
                self.data_processing_service[device_id] = Filter()
            
            lr_filtered = self.data_processing_service[device_id].filter(lr)

            if lr_filtered.status is LocationResultStatus.SUCCESS:
                logger.info(
                    "位置计算成功: (%.6f, %.6f), 方法: %s, 信标数: %s",
                    lr_filtered.latitude,
                    lr_filtered.longitude,
                    lr_filtered.method,
                    lr_filtered.beacon_count,
                )
                return lr_filtered
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
            logger.info("连接到MQTT服务器 %s:%s", mqtt_config["ip"], mqtt_config["port"])
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

    # ---------- MQTT handlers ----------
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("成功连接到MQTT服务器")
            mqtt_config = self.config_manager.get_mqtt_config()
            topic = mqtt_config.get("downlink_topic", "/device/blueTooth/station/+")
            client.subscribe(topic)
            self.current_topic = topic
            logger.info("已订阅主题: %s", topic)
        else:
            logger.error("连接失败，返回码: %s", rc)

    def on_message(self, client, userdata, msg: MQTTMessage):
        try:
            payload = msg.payload.decode("utf-8")
            with self.lock:
                record = BluetoothRecord.parse(payload)
                if record is None or record.is_empty:
                    logger.warning("消息解析无有效信标数据: %s", payload)
                    return
                location_data = self.calculate_location(record)
                if location_data:
                    # 转换为协议格式
                    mqtt_config = self.config_manager.get_mqtt_config()
                    topic = mqtt_config.get("uplink_topic", "/device/location/{deviceId}")
                    protocol_data = PositionProtocolData.from_location_result(location_data)
                    message = protocol_data.to_beidou_protocol_string()
                    self.client.publish(topic.format(deviceId=record.device_id), message)
        except Exception as e:
            logger.exception("处理消息时出错: %s", e)
