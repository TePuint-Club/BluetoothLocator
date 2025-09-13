from datetime import datetime
import paho.mqtt.client as mqtt
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import os
import math
import json
import time
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class ConfigManager:
    """配置管理类，负责读写YAML配置文件"""
    
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.default_config = {
            "mqtt": {
                "ip": "localhost",
                "port": 1883,
                "topic": "/device/blueTooth/station/+"
            },
            "rssi_model": {
                "tx_power": -59,  # 1米处的RSSI值 (dBm)
                "path_loss_exponent": 2.0  # 路径损失指数
            }
        }
        self.load_config()
    
    def load_config(self):
        """加载配置文件，如果不存在则创建默认配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                # 确保所有必要的键都存在
                self._merge_default_config()
            else:
                self.config = self.default_config.copy()
                self.save_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self.config = self.default_config.copy()
            self.save_config()
    
    def _merge_default_config(self):
        """合并默认配置，确保所有必要的键都存在"""
        def merge_dict(default, current):
            for key, value in default.items():
                if key not in current:
                    current[key] = value
                elif isinstance(value, dict) and isinstance(current[key], dict):
                    merge_dict(value, current[key])
        
        merge_dict(self.default_config, self.config)
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_mqtt_config(self):
        """获取MQTT配置"""
        return self.config["mqtt"]
    
    def get_rssi_model_config(self):
        """获取RSSI模型配置"""
        return self.config["rssi_model"]
    
    def set_mqtt_config(self, ip, port, topic=None):
        """设置MQTT配置"""
        self.config["mqtt"]["ip"] = ip
        self.config["mqtt"]["port"] = port
        if topic is not None:
            self.config["mqtt"]["topic"] = topic
        self.save_config()
    
    def set_rssi_model_config(self, tx_power, path_loss_exponent):
        """设置RSSI模型配置"""
        self.config["rssi_model"]["tx_power"] = tx_power
        self.config["rssi_model"]["path_loss_exponent"] = path_loss_exponent
        self.save_config()


# 全局配置管理器
config_manager = ConfigManager()

class BeaconLocationCalculator:
    """基于RSSI的蓝牙信标定位算法"""

    def __init__(self, config_manager=None):
        # 蓝牙信标位置数据库 (MAC地址 -> 位置信息)
        self.beacon_database = {}
        # 配置管理器
        self.config_manager = config_manager or ConfigManager()
        # RSSI-距离模型参数，从配置文件获取
        rssi_config = self.config_manager.get_rssi_model_config()
        self.tx_power = rssi_config["tx_power"]  # 1米处的RSSI值 (dBm)
        self.path_loss_exponent = rssi_config["path_loss_exponent"]  # 路径损失指数
        # 定位历史记录
        self.location_history = []
        self.location_csv_path = "terminal_locations.csv"
        self.init_location_csv()
    
    def update_rssi_model_params(self, tx_power, path_loss_exponent):
        """更新RSSI模型参数"""
        self.tx_power = tx_power
        self.path_loss_exponent = path_loss_exponent
        self.config_manager.set_rssi_model_config(tx_power, path_loss_exponent)

    def init_location_csv(self):
        """初始化位置记录CSV文件"""
        if not os.path.exists(self.location_csv_path):
            location_df = pd.DataFrame(columns=[
                "id", "device_id", "longitude", "latitude", "accuracy",
                "beacon_count", "timestamp", "calculation_method"
            ])
            location_df.to_csv(self.location_csv_path, index=False)

    def load_beacon_database(self, beacon_file_path="beacon_database.json"):
        """加载蓝牙信标位置数据库"""
        try:
            if os.path.exists(beacon_file_path):
                with open(beacon_file_path, 'r', encoding='utf-8') as f:
                    self.beacon_database = json.load(f)
            else:
                # 创建示例信标数据库
                self.create_sample_beacon_database(beacon_file_path)
        except Exception as e:
            print(f"加载信标数据库失败: {e}")
            self.create_sample_beacon_database(beacon_file_path)

    def create_sample_beacon_database(self, beacon_file_path="beacon_database.json"):
        """创建示例信标数据库"""
        sample_beacons = {
            "EXAMPLE-BEACON": {"longitude": 120, "latitude": 31, "altitude": 0.0},
        }
        self.beacon_database = sample_beacons
        try:
            with open(beacon_file_path, 'w', encoding='utf-8') as f:
                json.dump(sample_beacons, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存示例信标数据库失败: {e}")

    def add_beacon(self, mac_address, longitude, latitude, altitude=0.0):
        """添加信标到数据库"""
        self.beacon_database[mac_address] = {
            "longitude": longitude,
            "latitude": latitude,
            "altitude": altitude
        }
        self.save_beacon_database()

    def update_beacon(self, mac_address, longitude, latitude, altitude=0.0):
        """更新信标信息"""
        if mac_address in self.beacon_database:
            self.beacon_database[mac_address] = {
                "longitude": longitude,
                "latitude": latitude,
                "altitude": altitude
            }
            self.save_beacon_database()
            return True
        return False

    def delete_beacon(self, mac_address):
        """删除信标"""
        if mac_address in self.beacon_database:
            del self.beacon_database[mac_address]
            self.save_beacon_database()
            return True
        return False

    def get_all_beacons(self):
        """获取所有信标信息"""
        return dict(self.beacon_database)

    def save_beacon_database(self, beacon_file_path="beacon_database.json"):
        """保存信标数据库到文件"""
        try:
            with open(beacon_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.beacon_database, f,
                          indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存信标数据库失败: {e}")

    def rssi_to_distance(self, rssi, tx_power=None, path_loss_exponent=None):
        """
        基于RSSI计算距离 (单位: 米)
        使用路径损失模型: RSSI = TxPower - 10 * n * log10(d)
        """
        if tx_power is None:
            tx_power = self.tx_power
        if path_loss_exponent is None:
            path_loss_exponent = self.path_loss_exponent

        if rssi == 0:
            return -1.0

        ratio = tx_power * 1.0 / rssi
        if ratio < 1.0:
            return math.pow(ratio, 10)
        else:
            accuracy = (0.89976) * math.pow(ratio, 7.7095) + 0.111
            return accuracy

    def calculate_distance_improved(self, rssi):
        """改进的RSSI距离计算方法"""
        if rssi >= -50:
            return 0.5  # 很近
        elif rssi >= -70:
            return 1.0 + ((-50 - rssi) / 20) * 4  # 1-5米
        elif rssi >= -90:
            return 5.0 + ((-70 - rssi) / 20) * 10  # 5-15米
        else:
            return 15.0 + ((-90 - rssi) / 10) * 5  # 15米以上

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """计算两个经纬度点之间的距离（米）"""
        R = 6371000  # 地球半径，单位米

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def trilateration(self, beacon_positions, distances):
        """
        三边测量算法计算位置
        beacon_positions: [(lat1, lon1), (lat2, lon2), (lat3, lon3), ...]
        distances: [d1, d2, d3, ...]
        """
        if len(beacon_positions) < 3:
            return None

        def error_function(point):
            """计算误差函数"""
            lat, lon = point
            total_error = 0
            for i, (beacon_lat, beacon_lon) in enumerate(beacon_positions):
                calculated_distance = self.haversine_distance(
                    lat, lon, beacon_lat, beacon_lon)
                error = (calculated_distance - distances[i]) ** 2
                total_error += error
            return total_error

        # 使用质心作为初始猜测
        initial_lat = sum(pos[0]
                          for pos in beacon_positions) / len(beacon_positions)
        initial_lon = sum(pos[1]
                          for pos in beacon_positions) / len(beacon_positions)

        try:
            # 使用改进的梯度下降方法
            result = self.simple_minimize(
                error_function, [initial_lat, initial_lon])

            # 验证结果的合理性
            if result:
                result_lat, result_lon = result

                # 检查结果是否在合理范围内（距离信标不能太远）
                max_distance_to_beacons = 0
                for beacon_lat, beacon_lon in beacon_positions:
                    dist = self.haversine_distance(
                        result_lat, result_lon, beacon_lat, beacon_lon)
                    max_distance_to_beacons = max(
                        max_distance_to_beacons, dist)

                # 如果距离所有信标都很远（超过1000米），可能是计算错误
                if max_distance_to_beacons > 1000:
                    print(f"三边测量结果不合理，距离信标过远: {max_distance_to_beacons:.1f}米")
                    return None

                # 检查纬度和经度是否在合理范围内
                if not (-90 <= result_lat <= 90) or not (-180 <= result_lon <= 180):
                    print(f"三边测量结果超出地理坐标范围: ({result_lat}, {result_lon})")
                    return None

                return result
            else:
                return None

        except Exception as e:
            print(f"三边测量计算失败: {e}")
            return None

    def simple_minimize(self, func, initial_point, learning_rate=0.0001, max_iterations=1000, tolerance=1e-8):
        """改进的梯度下降优化算法"""
        x = list(initial_point)
        best_x = list(x)
        best_value = func(x)

        for iteration in range(max_iterations):
            # 计算数值梯度
            epsilon = 1e-8
            gradient = []

            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += epsilon
                x_minus[i] -= epsilon

                grad = (func(x_plus) - func(x_minus)) / (2 * epsilon)
                gradient.append(grad)

            # 计算梯度的模长
            grad_norm = sum(g*g for g in gradient) ** 0.5

            # 如果梯度很小，说明已经收敛
            if grad_norm < tolerance:
                break

            # 自适应学习率
            current_lr = learning_rate / (1 + iteration * 0.001)

            # 更新参数
            new_x = []
            for i in range(len(x)):
                new_x.append(x[i] - current_lr * gradient[i])

            # 检查新位置是否更好
            new_value = func(new_x)
            if new_value < best_value:
                best_value = new_value
                best_x = list(new_x)
                x = new_x
            else:
                # 如果没有改善，减小学习率
                learning_rate *= 0.5
                if learning_rate < 1e-10:
                    break

        return best_x

    def weighted_centroid(self, beacon_positions, rssi_values):
        """基于RSSI权重的质心算法"""
        if not beacon_positions:
            return None

        # 将RSSI转换为权重（RSSI越高，距离越近，权重越大）
        weights = []
        for rssi in rssi_values:
            # 将负的RSSI转换为正权重
            weight = max(0, rssi + 100)  # 假设最小RSSI为-100
            weights.append(weight)

        if sum(weights) == 0:
            # 如果所有权重都为0，使用简单平均
            lat = sum(pos[0]
                      for pos in beacon_positions) / len(beacon_positions)
            lon = sum(pos[1]
                      for pos in beacon_positions) / len(beacon_positions)
        else:
            # 计算加权平均
            total_weight = sum(weights)
            lat = sum(pos[0] * w for pos,
                      w in zip(beacon_positions, weights)) / total_weight
            lon = sum(pos[1] * w for pos,
                      w in zip(beacon_positions, weights)) / total_weight

        return [lat, lon]

    def calculate_terminal_location(self, bluetooth_readings):
        """
        根据蓝牙读数计算终端位置
        bluetooth_readings: [{"mac": "XX:XX:XX:XX:XX:XX", "rssi": -65}, ...]
        """
        if not bluetooth_readings:
            return None

        # 筛选数据库中存在的信标
        valid_readings = []
        beacon_positions = []
        distances = []
        rssi_values = []

        for reading in bluetooth_readings:
            mac = reading["mac"]
            rssi = reading["rssi"]

            if mac in self.beacon_database:
                beacon_info = self.beacon_database[mac]
                valid_readings.append(reading)
                beacon_positions.append(
                    [beacon_info["latitude"], beacon_info["longitude"]])
                distances.append(self.calculate_distance_improved(rssi))
                rssi_values.append(rssi)

        if len(valid_readings) == 0:
            return {
                "status": "error",
                "message": "没有找到已知位置的信标",
                "beacon_count": 0
            }
        elif len(valid_readings) == 1:
            # 只有一个信标，返回该信标位置
            beacon_pos = beacon_positions[0]
            return {
                "status": "single_beacon",
                "latitude": beacon_pos[0],
                "longitude": beacon_pos[1],
                "accuracy": distances[0],
                "beacon_count": 1,
                "method": "single_beacon"
            }
        elif len(valid_readings) == 2:
            # 两个信标，使用加权质心
            result = self.weighted_centroid(beacon_positions, rssi_values)
            if result:
                return {
                    "status": "success",
                    "latitude": result[0],
                    "longitude": result[1],
                    "accuracy": sum(distances) / len(distances),
                    "beacon_count": 2,
                    "method": "weighted_centroid"
                }
            else:
                return {
                    "status": "error",
                    "message": "加权质心计算失败",
                    "beacon_count": 2
                }
        else:
            # 三个或更多信标，使用三边测量
            result = self.trilateration(beacon_positions, distances)
            if result:
                # 计算精度估计
                accuracy = sum(distances) / len(distances)
                return {
                    "status": "success",
                    "latitude": result[0],
                    "longitude": result[1],
                    "accuracy": accuracy,
                    "beacon_count": len(valid_readings),
                    "method": "trilateration"
                }
            else:
                # 三边测量失败，使用加权质心作为备选
                result = self.weighted_centroid(beacon_positions, rssi_values)
                if result:
                    return {
                        "status": "fallback",
                        "latitude": result[0],
                        "longitude": result[1],
                        "accuracy": sum(distances) / len(distances),
                        "beacon_count": len(valid_readings),
                        "method": "weighted_centroid_fallback"
                    }
                else:
                    return {
                        "status": "error",
                        "message": "所有定位方法都失败",
                        "beacon_count": len(valid_readings)
                    }


class MQTTDataProcessor:
    def __init__(self, config_manager=None):
        self.bluetooth_data = []
        self.bluetooth_id_counter = 0
        self.location_id_counter = 0
        self.lock = threading.Lock()
        
        # 配置管理器
        self.config_manager = config_manager or ConfigManager()
        
        # 控制状态 - 默认暂停记录
        self.is_paused = True  # 默认暂停状态
        self.is_recording = False  # 记录状态
        
        # MQTT连接状态
        self.current_topic = None

        # 确保CSV文件存在
        self.bluetooth_csv_path = "bluetooth_position_data.csv"

        # 初始化定位计算器
        self.location_calculator = BeaconLocationCalculator(self.config_manager)
        self.location_calculator.load_beacon_database()

        # 创建CSV文件头部
        self.init_csv_files()

        # 消息队列用于GUI更新
        self.message_queue = None
        self.fn_message = None
        
        
    def on_location(self, fn_location):
        """处理位置数据"""
        assert callable(fn_location), "fn_location must be a callable function"
        self.fn_location = fn_location

    def on_gui_message(self, fn_message):
        assert callable(fn_message), "fn_message must be a callable function"
        self.fn_message = fn_message
    
    def pause_recording(self):
        """暂停数据记录"""
        with self.lock:
            self.is_paused = True
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 数据记录已暂停"
            print(message)
            self.fn_message(message) if self.fn_message else None
    
    def resume_recording(self):
        """恢复数据记录"""
        with self.lock:
            self.is_paused = False
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 数据记录已恢复"
            print(message)
            self.fn_message(message) if self.fn_message else None
    
    def stop_recording(self):
        """停止数据记录并重置计数器"""
        with self.lock:
            self.is_paused = True
            self.bluetooth_id_counter = 0
            self.location_id_counter = 0
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 数据记录已停止，计数器已重置"
            print(message)
            self.fn_message(message) if self.fn_message else None
    
    def get_recording_status(self):
        """获取当前记录状态"""
        return {
            "is_paused": self.is_paused,
            "is_recording": self.is_recording,
            "bluetooth_count": self.bluetooth_id_counter,
            "location_count": self.location_id_counter
        }
    
    def calculate_location_for_visualization(self, bluetooth_results):
        """计算位置用于可视化（不受记录状态影响）"""
        try:
            if not bluetooth_results:
                return
                
            # 准备蓝牙读数数据
            readings = []
            for result in bluetooth_results:
                readings.append({
                    "mac": result["mac"],
                    "rssi": result["rssi"]
                })

            # 计算位置
            location_result = self.location_calculator.calculate_terminal_location(readings)

            if location_result and location_result["status"] in ["success", "single_beacon", "fallback"]:
                # 准备可视化数据（包含设备ID）
                location_data = {
                    "device_id": bluetooth_results[0]["device_id"],
                    "longitude": location_result["longitude"],
                    "latitude": location_result["latitude"],
                    "accuracy": location_result["accuracy"],
                    "beacon_count": location_result["beacon_count"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calculation_method": location_result["method"]
                }

                # 传递位置数据给GUI用于可视化（始终执行）
                self.fn_location(location_data) if self.fn_location else None

        except Exception as e:
            print(f"可视化位置计算出错: {str(e)}")

    def init_csv_files(self):
        # 初始化蓝牙数据CSV文件
        if not os.path.exists(self.bluetooth_csv_path):
            bluetooth_df = pd.DataFrame(
                columns=["id", "device_id", "mac", "rssi", "rotation", "timestamp"])
            bluetooth_df.to_csv(self.bluetooth_csv_path, index=False)

    def handle_bluetooth_position_data(self, data_str: str) -> list[dict]:
        data = data_str.split(";")
        results = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device_id = data[-1]
        for item in data[:-1]:
            mac, rssi, rotation = item.split(",")
            results.append({
                "device_id": device_id,
                "mac": mac,
                "rssi": int(rssi),
                "rotation": int(rotation),
                "timestamp": current_time,
            })
        return results

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 成功连接到MQTT服务器"
            print(message)
            self.fn_message(message) if self.fn_message else None
                
            # 订阅主题
            mqtt_config = self.config_manager.get_mqtt_config()
            topic = mqtt_config.get("topic", "/device/blueTooth/station/+")
            client.subscribe(topic)
            self.current_topic = topic

            message = f"[{datetime.now().strftime('%H:%M:%S')}] 已订阅主题: {topic}"
            print(message)
            self.fn_message(message) if self.fn_message else None
        else:
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 连接失败，返回码: {rc}"
            print(message)
            self.fn_message(message) if self.fn_message else None

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 收到消息 - 主题: {topic}, 内容: {payload}"
            self.fn_message(message) if self.fn_message else None
            print(message)

            with self.lock:
                # 处理蓝牙数据（用于可视化，不受暂停状态影响）
                bluetooth_results = self.handle_bluetooth_position_data(
                    payload)
                
                # 始终尝试计算位置用于可视化
                self.calculate_location_for_visualization(bluetooth_results)
                
                # 检查是否暂停记录
                if self.is_paused:
                    message = f"[{datetime.now().strftime('%H:%M:%S')}] 因暂停跳过数据记录 - 设备ID: {bluetooth_results[0]['device_id'] if bluetooth_results else 'Unknown'}"
                    print(message)
                    self.fn_message(message) if self.fn_message else None
                    return
                
                # 为每个蓝牙数据项添加ID
                for result in bluetooth_results:
                    result["id"] = self.bluetooth_id_counter
                    self.bluetooth_data.append(result)

                self.bluetooth_id_counter += 1

                # 保存到CSV
                self.save_bluetooth_data_to_csv(bluetooth_results)

                # 计算并保存终端位置
                self.calculate_and_save_location(bluetooth_results)

                message = f"[{datetime.now().strftime('%H:%M:%S')}] 蓝牙数据已处理并记录，当前ID: {self.bluetooth_id_counter-1}, 设备: {bluetooth_results[0]['device_id']}"
                print(message)
                self.fn_message(message) if self.fn_message else None

        except Exception as e:
            error_message = f"[{datetime.now().strftime('%H:%M:%S')}] 处理消息时出错: {str(e)}"
            print(error_message)
            self.fn_message(error_message) if self.fn_message else None

    def save_bluetooth_data_to_csv(self, new_data):
        """实时保存蓝牙数据到CSV"""
        df_new = pd.DataFrame(new_data)
        # 确保列的顺序正确
        df_new = df_new[["id", "device_id", "mac",
                         "rssi", "rotation", "timestamp"]]
        df_new.to_csv(self.bluetooth_csv_path, mode='a',
                      header=False, index=False)

    def calculate_and_save_location(self, bluetooth_results):
        """计算并保存终端位置"""
        try:
            # 准备蓝牙读数数据
            readings = []
            for result in bluetooth_results:
                readings.append({
                    "mac": result["mac"],
                    "rssi": result["rssi"]
                })

            # 计算位置
            location_result = self.location_calculator.calculate_terminal_location(
                readings)

            if location_result and location_result["status"] in ["success", "single_beacon", "fallback"]:
                # 准备保存数据
                location_data = {
                    "id": self.location_id_counter,
                    "device_id": bluetooth_results[0]["device_id"],
                    "longitude": location_result["longitude"],
                    "latitude": location_result["latitude"],
                    "accuracy": location_result["accuracy"],
                    "beacon_count": location_result["beacon_count"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calculation_method": location_result["method"]
                }

                # 保存到CSV
                self.save_location_to_csv(location_data)

                # 传递位置数据给GUI用于可视化
                self.fn_location(location_data) if self.fn_location else None

                message = f"[{datetime.now().strftime('%H:%M:%S')}] 位置计算成功: ({location_result['latitude']:.6f}, {location_result['longitude']:.6f}), 方法: {location_result['method']}, 信标数: {location_result['beacon_count']}"
                print(message)
                self.fn_message(message) if self.fn_message else None

                self.location_id_counter += 1
            else:
                message = f"[{datetime.now().strftime('%H:%M:%S')}] 位置计算失败: {location_result.get('message', '未知错误') if location_result else '计算返回None'}"
                print(message)
                self.fn_message(message) if self.fn_message else None

        except Exception as e:
            error_message = f"[{datetime.now().strftime('%H:%M:%S')}] 位置计算出错: {str(e)}"
            print(error_message)
            self.fn_message(message) if self.fn_message else None

    def save_location_to_csv(self, location_data):
        """保存位置数据到CSV"""
        df_new = pd.DataFrame([location_data])
        column_order = [
            "id", "device_id", "longitude", "latitude", "accuracy",
            "beacon_count", "timestamp", "calculation_method"
        ]
        df_new = df_new[column_order]
        df_new.to_csv(self.location_calculator.location_csv_path,
                      mode='a', header=False, index=False)

    def start_mqtt_client(self):
        """启动MQTT客户端"""
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            mqtt_config = self.config_manager.get_mqtt_config()
            self.client.connect(mqtt_config["ip"], mqtt_config["port"], 60)
            self.client.loop_forever()
        except Exception as e:
            error_message = f"[{datetime.now().strftime('%H:%M:%S')}] MQTT连接错误: {str(e)}"
            self.fn_message(error_message) if self.fn_message else None
    
    def stop_mqtt_client(self):
        """停止MQTT客户端"""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.disconnect()
                self.client.loop_stop()
                message = f"[{datetime.now().strftime('%H:%M:%S')}] MQTT连接已断开"
                print(message)
                self.fn_message(message) if self.fn_message else None
            except Exception as e:
                error_message = f"[{datetime.now().strftime('%H:%M:%S')}] 断开MQTT连接时出错: {str(e)}"
                self.fn_message(error_message) if self.fn_message else None
    
    def change_mqtt_topic(self, new_topic):
        """更改MQTT主题订阅"""
        if hasattr(self, 'client') and self.client and self.current_topic:
            try:
                # 取消订阅当前主题
                self.client.unsubscribe(self.current_topic)
                message = f"[{datetime.now().strftime('%H:%M:%S')}] 已取消订阅主题: {self.current_topic}"
                print(message)
                self.fn_message(message) if self.fn_message else None
                
                # 订阅新主题
                self.client.subscribe(new_topic)
                self.current_topic = new_topic
                message = f"[{datetime.now().strftime('%H:%M:%S')}] 已订阅新主题: {new_topic}"
                print(message)
                self.fn_message(message) if self.fn_message else None
                
                return True
            except Exception as e:
                error_message = f"[{datetime.now().strftime('%H:%M:%S')}] 更改主题订阅时出错: {str(e)}"
                self.fn_message(error_message) if self.fn_message else None
                return False
        return False


class DataMonitorGUI:
    def __init__(self, processor: MQTTDataProcessor, config_manager: ConfigManager):
        self.processor = processor
        self.config_manager = config_manager
        self.message_queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title("蓝牙信标定位监控系统")
        self.root.geometry("1200x800")

        # 位置历史记录 - 支持多设备
        self.location_history = {}  # 改为字典，key为device_id
        
        # MQTT客户端管理
        self.mqtt_client = None
        self.mqtt_thread = None

        self.setup_gui()

    def setup_gui(self):
        # 创建主选项卡控件
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建各个选项卡
        self.create_monitor_tab()
        self.create_beacon_management_tab()
        self.create_visualization_tab()
        self.create_settings_tab()

        # 启动更新线程
        self.update_gui()

    def create_monitor_tab(self):
        """创建数据监控选项卡"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="数据监控")

        # 主框架
        main_frame = ttk.Frame(monitor_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 状态标签
        status_frame = ttk.LabelFrame(main_frame, text="数据统计", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.bluetooth_label = ttk.Label(status_frame, text="蓝牙数据处理数量: 0")
        self.bluetooth_label.pack(anchor=tk.W, pady=2)

        self.location_label = ttk.Label(status_frame, text="位置计算数量: 0")
        self.location_label.pack(anchor=tk.W, pady=2)

        self.status_label = ttk.Label(status_frame, text="状态: 已暂停", foreground="red")
        self.status_label.pack(anchor=tk.W, pady=2)

        # 消息日志
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=20, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # 第一行按钮
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=(0, 5))

        self.start_button = ttk.Button(
            button_row1, text="启动MQTT监听", command=self.start_mqtt)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.reconnect_button = ttk.Button(
            button_row1, text="重新连接", command=self.reconnect_mqtt, state="disabled")
        self.reconnect_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = ttk.Button(
            button_row1, text="清空日志", command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT)

        # 第二行按钮 - 记录控制
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(fill=tk.X)

        self.pause_button = ttk.Button(
            button_row2, text="暂停记录", command=self.pause_recording, state="disabled")
        self.pause_button.pack(side=tk.LEFT, padx=(0, 10))

        self.resume_button = ttk.Button(
            button_row2, text="恢复记录", command=self.resume_recording)
        self.resume_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(
            button_row2, text="停止记录", command=self.stop_recording)
        self.stop_button.pack(side=tk.LEFT)

    def create_beacon_management_tab(self):
        """创建信标管理选项卡"""
        beacon_frame = ttk.Frame(self.notebook)
        self.notebook.add(beacon_frame, text="信标管理")

        # 主框架
        main_frame = ttk.Frame(beacon_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 信标列表框架
        list_frame = ttk.LabelFrame(main_frame, text="信标列表", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 创建Treeview来显示信标
        columns = ("MAC地址", "经度", "纬度", "高度")
        self.beacon_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=15)

        # 设置列标题
        for col in columns:
            self.beacon_tree.heading(col, text=col)
            self.beacon_tree.column(col, width=150)

        # 添加滚动条
        beacon_scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.beacon_tree.yview)
        self.beacon_tree.configure(yscrollcommand=beacon_scrollbar.set)

        self.beacon_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        beacon_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 信标操作框架
        operation_frame = ttk.LabelFrame(main_frame, text="信标操作", padding="10")
        operation_frame.pack(fill=tk.X)

        # 输入框架
        input_frame = ttk.Frame(operation_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        # MAC地址输入
        ttk.Label(input_frame, text="MAC地址:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.mac_entry = ttk.Entry(input_frame, width=20)
        self.mac_entry.grid(row=0, column=1, padx=(0, 10))

        # 经度输入
        ttk.Label(input_frame, text="经度:").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.lon_entry = ttk.Entry(input_frame, width=15)
        self.lon_entry.grid(row=0, column=3, padx=(0, 10))

        # 纬度输入
        ttk.Label(input_frame, text="纬度:").grid(
            row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.lat_entry = ttk.Entry(input_frame, width=15)
        self.lat_entry.grid(row=0, column=5, padx=(0, 10))

        # 高度输入
        ttk.Label(input_frame, text="高度:").grid(
            row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.alt_entry = ttk.Entry(input_frame, width=10)
        self.alt_entry.grid(row=0, column=7)

        # 按钮框架
        button_frame = ttk.Frame(operation_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="添加信标", command=self.add_beacon).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="修改信标", command=self.update_beacon).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="删除信标", command=self.delete_beacon).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="刷新列表", command=self.refresh_beacon_list).pack(
            side=tk.LEFT, padx=(0, 5))

        # 绑定选择事件
        self.beacon_tree.bind("<<TreeviewSelect>>", self.on_beacon_select)

        # 初始化信标列表
        self.root.after(100, self.refresh_beacon_list)

    def create_visualization_tab(self):
        """创建可视化选项卡"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="位置可视化")

        # 主框架
        main_frame = ttk.Frame(viz_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 控制框架
        control_frame = ttk.LabelFrame(main_frame, text="可视化控制", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.auto_update_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="自动更新", variable=self.auto_update_var).pack(
            side=tk.LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="手动刷新", command=self.update_visualization).pack(
            side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="清空历史",
                   command=self.clear_location_history).pack(side=tk.LEFT)

        # 创建matplotlib图形
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 创建Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始化图形
        self.init_plot()

    def init_plot(self):
        """初始化绘图"""
        self.ax.clear()
        self.ax.set_title("蓝牙信标定位可视化")
        self.ax.set_xlabel("经度")
        self.ax.set_ylabel("纬度")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def update_visualization(self):
        """更新可视化图形"""
        if not self.processor:
            return

        try:
            self.ax.clear()

            # 获取信标位置
            beacons = self.processor.location_calculator.get_all_beacons()

            if beacons:
                # 绘制信标
                beacon_lons = [info['longitude'] for info in beacons.values()]
                beacon_lats = [info['latitude'] for info in beacons.values()]

                self.ax.scatter(beacon_lons, beacon_lats, c='blue', s=100, marker='^',
                                label='信标位置', alpha=0.8, edgecolors='darkblue')

                # 添加信标标签
                for mac, info in beacons.items():
                    self.ax.annotate(mac[-4:], (info['longitude'], info['latitude']),
                                     xytext=(5, 5), textcoords='offset points', fontsize=8)

                # 如果有位置历史记录，绘制多设备轨迹
                colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                all_lons = beacon_lons.copy()
                all_lats = beacon_lats.copy()
                
                color_index = 0
                for device_id, device_history in self.location_history.items():
                    if not device_history:
                        continue
                        
                    color = colors[color_index % len(colors)]
                    color_index += 1
                    
                    # 绘制该设备的历史轨迹
                    device_lons = [loc['longitude'] for loc in device_history]
                    device_lats = [loc['latitude'] for loc in device_history]
                    
                    all_lons.extend(device_lons)
                    all_lats.extend(device_lats)

                    if len(device_lons) > 1:
                        self.ax.plot(device_lons, device_lats, color=color,
                                     alpha=0.6, linewidth=2, label=f'设备{device_id}轨迹')

                    # 绘制该设备的当前位置
                    if device_history:
                        current = device_history[-1]
                        self.ax.scatter([current['longitude']], [current['latitude']],
                                        c=color, s=150, marker='o', 
                                        alpha=0.9, edgecolors='black', linewidth=2)
                        
                        # 添加设备ID标签
                        self.ax.annotate(f'设备{device_id}', 
                                        (current['longitude'], current['latitude']),
                                        xytext=(10, 10), textcoords='offset points', 
                                        fontsize=10, fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

                # 设置合适的显示范围
                if all_lons and all_lats:
                    lon_margin = (max(all_lons) - min(all_lons)) * 0.1 or 0.001
                    lat_margin = (max(all_lats) - min(all_lats)) * 0.1 or 0.001

                    self.ax.set_xlim(min(all_lons) - lon_margin,
                                     max(all_lons) + lon_margin)
                    self.ax.set_ylim(min(all_lats) - lat_margin,
                                     max(all_lats) + lat_margin)

            self.ax.set_title("蓝牙信标定位可视化")
            self.ax.set_xlabel("经度")
            self.ax.set_ylabel("纬度")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()

            self.canvas.draw()

        except Exception as e:
            print(f"更新可视化时出错: {e}")

    def create_settings_tab(self):
        """创建设置选项卡"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="系统设置")

        # 主框架
        main_frame = ttk.Frame(settings_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # MQTT设置框架
        mqtt_frame = ttk.LabelFrame(main_frame, text="MQTT服务器设置", padding="10")
        mqtt_frame.pack(fill=tk.X, pady=(0, 10))

        # IP地址设置
        ip_frame = ttk.Frame(mqtt_frame)
        ip_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(ip_frame, text="IP地址:").pack(side=tk.LEFT, padx=(0, 5))
        self.mqtt_ip_entry = ttk.Entry(ip_frame, width=20)
        self.mqtt_ip_entry.pack(side=tk.LEFT, padx=(0, 10))

        # 端口设置
        port_frame = ttk.Frame(mqtt_frame)
        port_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(port_frame, text="端口:").pack(side=tk.LEFT, padx=(0, 5))
        self.mqtt_port_entry = ttk.Entry(port_frame, width=10)
        self.mqtt_port_entry.pack(side=tk.LEFT, padx=(0, 10))

        # 订阅主题设置
        topic_frame = ttk.Frame(mqtt_frame)
        topic_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(topic_frame, text="订阅主题:").pack(side=tk.LEFT, padx=(0, 5))
        self.mqtt_topic_entry = ttk.Entry(topic_frame, width=30)
        self.mqtt_topic_entry.pack(side=tk.LEFT, padx=(0, 10))

        # 常用主题快速选择
        topic_buttons_frame = ttk.Frame(mqtt_frame)
        topic_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(topic_buttons_frame, text="常用主题:").pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(topic_buttons_frame, text="默认", 
                   command=lambda: self.set_topic("/device/blueTooth/station/+")).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(topic_buttons_frame, text="全部", 
                   command=lambda: self.set_topic("#")).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(topic_buttons_frame, text="测试", 
                   command=lambda: self.set_topic("test/topic")).pack(side=tk.LEFT, padx=(0, 2))

        # MQTT按钮
        mqtt_button_frame = ttk.Frame(mqtt_frame)
        mqtt_button_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(mqtt_button_frame, text="保存MQTT设置", command=self.save_mqtt_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(mqtt_button_frame, text="恢复默认", command=self.reset_mqtt_settings).pack(side=tk.LEFT)

        # RSSI模型设置框架
        rssi_frame = ttk.LabelFrame(main_frame, text="RSSI-距离模型参数", padding="10")
        rssi_frame.pack(fill=tk.X, pady=(0, 10))

        # 1米处RSSI值设置
        tx_power_frame = ttk.Frame(rssi_frame)
        tx_power_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(tx_power_frame, text="1米处的RSSI值 (dBm):").pack(side=tk.LEFT, padx=(0, 5))
        self.tx_power_entry = ttk.Entry(tx_power_frame, width=10)
        self.tx_power_entry.pack(side=tk.LEFT, padx=(0, 10))

        # 路径损失指数设置
        path_loss_frame = ttk.Frame(rssi_frame)
        path_loss_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(path_loss_frame, text="路径损失指数:").pack(side=tk.LEFT, padx=(0, 5))
        self.path_loss_entry = ttk.Entry(path_loss_frame, width=10)
        self.path_loss_entry.pack(side=tk.LEFT, padx=(0, 10))

        # RSSI模型按钮
        rssi_button_frame = ttk.Frame(rssi_frame)
        rssi_button_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(rssi_button_frame, text="保存RSSI设置", command=self.save_rssi_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(rssi_button_frame, text="恢复默认", command=self.reset_rssi_settings).pack(side=tk.LEFT)

        # 配置信息显示框架
        info_frame = ttk.LabelFrame(main_frame, text="当前配置信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.config_info_text = scrolledtext.ScrolledText(info_frame, height=10, width=60)
        self.config_info_text.pack(fill=tk.BOTH, expand=True)

        # 加载当前设置
        self.load_current_settings()

    def load_current_settings(self):
        """加载当前配置到输入框"""
        if not self.config_manager:
            return

        # 加载MQTT设置
        mqtt_config = self.config_manager.get_mqtt_config()
        self.mqtt_ip_entry.delete(0, tk.END)
        self.mqtt_ip_entry.insert(0, mqtt_config["ip"])
        self.mqtt_port_entry.delete(0, tk.END)
        self.mqtt_port_entry.insert(0, str(mqtt_config["port"]))
        self.mqtt_topic_entry.delete(0, tk.END)
        self.mqtt_topic_entry.insert(0, mqtt_config.get("topic", "/device/blueTooth/station/+"))

        # 加载RSSI模型设置
        rssi_config = self.config_manager.get_rssi_model_config()
        self.tx_power_entry.delete(0, tk.END)
        self.tx_power_entry.insert(0, str(rssi_config["tx_power"]))
        self.path_loss_entry.delete(0, tk.END)
        self.path_loss_entry.insert(0, str(rssi_config["path_loss_exponent"]))

        # 更新配置信息显示
        self.update_config_info_display()

    def set_topic(self, topic):
        """设置主题到输入框"""
        self.mqtt_topic_entry.delete(0, tk.END)
        self.mqtt_topic_entry.insert(0, topic)

    def save_mqtt_settings(self):
        """保存MQTT设置"""
        try:
            ip = self.mqtt_ip_entry.get().strip()
            port = int(self.mqtt_port_entry.get())
            topic = self.mqtt_topic_entry.get().strip()

            if not ip:
                messagebox.showerror("错误", "请输入IP地址")
                return

            if port <= 0 or port > 65535:
                messagebox.showerror("错误", "端口号必须在1-65535之间")
                return

            if not topic:
                messagebox.showerror("错误", "请输入订阅主题")
                return

            # 检查是否只是主题发生了变化
            current_config = self.config_manager.get_mqtt_config()
            topic_changed = current_config.get("topic") != topic
            connection_changed = (current_config.get("ip") != ip or 
                                current_config.get("port") != port)

            self.config_manager.set_mqtt_config(ip, port, topic)
            self.update_config_info_display()

            if topic_changed and not connection_changed:
                # 只有主题变化，尝试只更换主题订阅
                if self.processor and hasattr(self.processor, 'current_topic') and self.processor.current_topic:
                    if self.processor.change_mqtt_topic(topic):
                        messagebox.showinfo("成功", "MQTT主题设置已保存并应用")
                        return
                
            # 需要重新连接
            messagebox.showinfo("成功", "MQTT设置已保存\n注意：需要重新连接MQTT服务器才能生效")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")

    def reset_mqtt_settings(self):
        """恢复MQTT默认设置"""
        self.mqtt_ip_entry.delete(0, tk.END)
        self.mqtt_ip_entry.insert(0, "localhost")
        self.mqtt_port_entry.delete(0, tk.END)
        self.mqtt_port_entry.insert(0, "1883")
        self.mqtt_topic_entry.delete(0, tk.END)
        self.mqtt_topic_entry.insert(0, "/device/blueTooth/station/+")

    def save_rssi_settings(self):
        """保存RSSI模型设置"""
        try:
            tx_power = float(self.tx_power_entry.get())
            path_loss_exponent = float(self.path_loss_entry.get())

            if path_loss_exponent <= 0:
                messagebox.showerror("错误", "路径损失指数必须大于0")
                return

            self.config_manager.set_rssi_model_config(tx_power, path_loss_exponent)
            
            # 更新处理器中的参数
            if self.processor:
                self.processor.location_calculator.update_rssi_model_params(tx_power, path_loss_exponent)
            
            self.update_config_info_display()
            messagebox.showinfo("成功", "RSSI模型设置已保存")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")

    def reset_rssi_settings(self):
        """恢复RSSI模型默认设置"""
        self.tx_power_entry.delete(0, tk.END)
        self.tx_power_entry.insert(0, "-59")
        self.path_loss_entry.delete(0, tk.END)
        self.path_loss_entry.insert(0, "2.0")

    def update_config_info_display(self):
        """更新配置信息显示"""
        if not self.config_manager:
            return

        try:
            config_text = "当前配置:\n\n"
            
            # MQTT配置
            mqtt_config = self.config_manager.get_mqtt_config()
            config_text += "MQTT服务器配置:\n"
            config_text += f"  IP地址: {mqtt_config['ip']}\n"
            config_text += f"  端口: {mqtt_config['port']}\n"
            config_text += f"  订阅主题: {mqtt_config.get('topic', '/device/blueTooth/station/+')}\n\n"
            
            # RSSI模型配置
            rssi_config = self.config_manager.get_rssi_model_config()
            config_text += "RSSI-距离模型参数:\n"
            config_text += f"  1米处的RSSI值: {rssi_config['tx_power']} dBm\n"
            config_text += f"  路径损失指数: {rssi_config['path_loss_exponent']}\n\n"
            
            # 配置文件路径
            config_text += f"配置文件路径: {self.config_manager.config_file}\n"

            self.config_info_text.delete(1.0, tk.END)
            self.config_info_text.insert(1.0, config_text)

        except Exception as e:
            print(f"更新配置信息显示时出错: {e}")

    def add_location_to_history(self, location_data):
        """添加位置数据到历史记录"""
        device_id = location_data.get('device_id', 'Unknown')
        
        # 如果该设备ID不存在，创建新的历史记录列表
        if device_id not in self.location_history:
            self.location_history[device_id] = []
        
        # 添加位置数据
        self.location_history[device_id].append({
            'longitude': location_data['longitude'],
            'latitude': location_data['latitude'],
            'timestamp': location_data['timestamp'],
            'accuracy': location_data['accuracy'],
            'method': location_data['calculation_method'],
            'device_id': device_id
        })

        # 限制每个设备的历史记录数量（保留最近100个位置）
        if len(self.location_history[device_id]) > 100:
            self.location_history[device_id].pop(0)

    def clear_location_history(self):
        """清空位置历史"""
        self.location_history.clear()
        self.update_visualization()

    def refresh_beacon_list(self):
        """刷新信标列表"""
        if not self.processor:
            return

        # 清空现有项目
        for item in self.beacon_tree.get_children():
            self.beacon_tree.delete(item)

        # 添加信标信息
        beacons = self.processor.location_calculator.get_all_beacons()
        for mac, info in beacons.items():
            self.beacon_tree.insert("", tk.END, values=(
                mac,
                f"{info['longitude']:.6f}",
                f"{info['latitude']:.6f}",
                f"{info['altitude']:.2f}"
            ))

    def on_beacon_select(self, event):
        """当选择信标时，填充到输入框"""
        selection = self.beacon_tree.selection()
        if selection:
            item = self.beacon_tree.item(selection[0])
            values = item['values']

            self.mac_entry.delete(0, tk.END)
            self.mac_entry.insert(0, values[0])

            self.lon_entry.delete(0, tk.END)
            self.lon_entry.insert(0, values[1])

            self.lat_entry.delete(0, tk.END)
            self.lat_entry.insert(0, values[2])

            self.alt_entry.delete(0, tk.END)
            self.alt_entry.insert(0, values[3])

    def add_beacon(self):
        """添加新信标"""
        if not self.processor:
            messagebox.showerror("错误", "处理器未初始化")
            return

        try:
            mac = self.mac_entry.get().strip().upper()
            lon = float(self.lon_entry.get())
            lat = float(self.lat_entry.get())
            alt = float(self.alt_entry.get()) if self.alt_entry.get() else 0.0

            if not mac:
                messagebox.showerror("错误", "请输入MAC地址")
                return

            self.processor.location_calculator.add_beacon(mac, lon, lat, alt)
            self.refresh_beacon_list()
            self.clear_entries()
            messagebox.showinfo("成功", "信标添加成功")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
        except Exception as e:
            messagebox.showerror("错误", f"添加信标失败: {e}")

    def update_beacon(self):
        """更新信标信息"""
        if not self.processor:
            messagebox.showerror("错误", "处理器未初始化")
            return

        try:
            mac = self.mac_entry.get().strip().upper()
            lon = float(self.lon_entry.get())
            lat = float(self.lat_entry.get())
            alt = float(self.alt_entry.get()) if self.alt_entry.get() else 0.0

            if not mac:
                messagebox.showerror("错误", "请输入MAC地址")
                return

            if self.processor.location_calculator.update_beacon(mac, lon, lat, alt):
                self.refresh_beacon_list()
                self.clear_entries()
                messagebox.showinfo("成功", "信标更新成功")
            else:
                messagebox.showerror("错误", "信标不存在")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
        except Exception as e:
            messagebox.showerror("错误", f"更新信标失败: {e}")

    def delete_beacon(self):
        """删除信标"""
        if not self.processor:
            messagebox.showerror("错误", "处理器未初始化")
            return

        try:
            mac = self.mac_entry.get().strip().upper()

            if not mac:
                messagebox.showerror("错误", "请输入要删除的MAC地址")
                return

            if messagebox.askyesno("确认", f"确定要删除信标 {mac} 吗？"):
                if self.processor.location_calculator.delete_beacon(mac):
                    self.refresh_beacon_list()
                    self.clear_entries()
                    messagebox.showinfo("成功", "信标删除成功")
                else:
                    messagebox.showerror("错误", "信标不存在")

        except Exception as e:
            messagebox.showerror("错误", f"删除信标失败: {e}")

    def clear_entries(self):
        """清空输入框"""
        self.mac_entry.delete(0, tk.END)
        self.lon_entry.delete(0, tk.END)
        self.lat_entry.delete(0, tk.END)
        self.alt_entry.delete(0, tk.END)

    def pause_recording(self):
        """暂停数据记录"""
        if self.processor:
            self.processor.pause_recording()
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="normal")
    
    def resume_recording(self):
        """恢复数据记录"""
        if self.processor:
            self.processor.resume_recording()
            self.pause_button.config(state="normal")
            self.resume_button.config(state="disabled")
    
    def stop_recording(self):
        """停止数据记录"""
        if self.processor:
            result = messagebox.askyesno("确认", "确定要停止数据记录并重置计数器吗？\n注意：这将重置ID计数器，但不会删除已有数据。")
            if result:
                self.processor.stop_recording()
                # 停止后设置为暂停状态的按钮状态
                self.pause_button.config(state="disabled")
                self.resume_button.config(state="normal")

    def start_mqtt(self):
        if self.processor:
            self.start_button.config(state="disabled")
            self.reconnect_button.config(state="normal")
            self.mqtt_thread = threading.Thread(
                target=self.processor.start_mqtt_client, daemon=True)
            self.mqtt_thread.start()
    
    def reconnect_mqtt(self):
        """重新连接MQTT服务器"""
        if self.processor:
            try:
                # 先停止当前连接
                self.processor.stop_mqtt_client()
                
                # 等待短暂时间确保连接完全断开
                time.sleep(1)
                
                # 启动新连接
                self.mqtt_thread = threading.Thread(
                    target=self.processor.start_mqtt_client, daemon=True)
                self.mqtt_thread.start()
                
                message = f"[{datetime.now().strftime('%H:%M:%S')}] 正在重新连接MQTT服务器..."
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
            except Exception as e:
                error_message = f"[{datetime.now().strftime('%H:%M:%S')}] 重连失败: {str(e)}"
                self.log_text.insert(tk.END, error_message + "\n")
                self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def update_gui(self):
        if not self.processor:
            self.root.after(100, self.update_gui)
            return

        # 更新状态标签
        with self.processor.lock:
            bluetooth_count = self.processor.bluetooth_id_counter
            location_count = self.processor.location_id_counter
            is_paused = self.processor.is_paused

        self.bluetooth_label.config(text=f"蓝牙数据处理数量: {bluetooth_count}")
        self.location_label.config(text=f"位置计算数量: {location_count}")
        
        # 更新状态显示
        if is_paused:
            self.status_label.config(text="状态: 已暂停", foreground="red")
        else:
            self.status_label.config(text="状态: 正在记录", foreground="green")

        # 更新日志
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass

        # 自动更新可视化
        if hasattr(self, 'auto_update_var') and self.auto_update_var.get():
            try:
                self.update_visualization()
            except:
                pass  # 忽略可视化更新错误

        # 每100ms更新一次
        self.root.after(100, self.update_gui)

    def run(self):
        self.root.mainloop()


def main():
    # 创建配置管理器
    config_mgr = config_manager
    
    # 检查配置是否有效
    mqtt_config = config_mgr.get_mqtt_config()
    if not mqtt_config["ip"] or mqtt_config["ip"] == "*#*#not_a_real_ip#*#*":
        print("请在设置中配置正确的MQTT服务器IP地址")
    
    # 创建数据处理器并传入配置管理器
    processor = MQTTDataProcessor(config_mgr)
    # 创建GUI并传入配置管理器
    gui = DataMonitorGUI(processor, config_mgr)
    processor.on_location(gui.add_location_to_history)  # 传递位置更新函数
    processor.on_gui_message(gui.message_queue.put)  # 传递日志更新函数

    # 运行GUI
    gui.run()


if __name__ == "__main__":
    main()
