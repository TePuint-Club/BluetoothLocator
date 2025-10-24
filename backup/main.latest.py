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
from tkinter import filedialog
import numpy as np
from copy import deepcopy

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


BEACON_DB_PATH = "./beacon/used.json"
LOCATION_CSV_PATH = "./output/terminal_locations.csv"
DATA_PATH_JSON_PATH = "./config/bluetooth_data.json"
STRATEGY1 = 'abandon' # 选择策略1: 'abandon'抛弃掉只有一个点的数据
STRATEGY2 = 'all' # 所有情况使用一种方法
FILTERRAW = False
FILTERCOMPARE = False
METHODCOMPARE = True

class ConfigManager:
    """配置管理类，负责读写YAML配置文件"""
    
    def __init__(self, config_file="./config/config.yaml"):
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
            },
            "optimization": {
                "use_multi_start": True,  # 是否使用多初始点优化
                "num_starts": 20,  # 初始点数量
                "search_radius": 0.001  # 搜索半径（度）
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
    
    def set_rssi_model_config(self, tx_power, path_loss_exponent,a,b):
        """设置RSSI模型配置"""
        self.config["rssi_model"]["tx_power"] = tx_power
        self.config["rssi_model"]["path_loss_exponent"] = path_loss_exponent
        self.config["rssi_model"]["a"] = a
        self.config["rssi_model"]["b"] = b
        self.save_config()
    
    def get_optimization_config(self):
        """获取优化算法配置"""
        return self.config.get("optimization", {
            "use_multi_start": True,
            "num_starts": 10,
            "search_radius": 0.001
        })
    
    def set_optimization_config(self, use_multi_start=True, num_starts=10, search_radius=0.001):
        """设置优化算法配置"""
        if "optimization" not in self.config:
            self.config["optimization"] = {}
        self.config["optimization"]["use_multi_start"] = use_multi_start
        self.config["optimization"]["num_starts"] = num_starts
        self.config["optimization"]["search_radius"] = search_radius
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
        self.location_csv_path = LOCATION_CSV_PATH
        self.init_location_csv()
    
    def update_rssi_model_params(self, tx_power, path_loss_exponent,a,b):
        """更新RSSI模型参数"""
        self.tx_power = tx_power
        self.path_loss_exponent = path_loss_exponent
        self.a = a
        self.b = b
        self.config_manager.set_rssi_model_config(tx_power, path_loss_exponent,a,b)

    def init_location_csv(self):
        """初始化位置记录CSV文件"""
        os.makedirs(os.path.dirname(self.location_csv_path), exist_ok=True)
        if not os.path.exists(self.location_csv_path):
            location_df = pd.DataFrame(columns=[
                "id", "device_id", "longitude", "latitude", "accuracy",
                "beacon_count", "timestamp", "calculation_method"
            ])
            location_df.to_csv(self.location_csv_path, index=False)

    def load_beacon_database(self, beacon_file_path=BEACON_DB_PATH):
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

    def create_sample_beacon_database(self, beacon_file_path=BEACON_DB_PATH):
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

    def save_beacon_database(self, beacon_file_path=BEACON_DB_PATH):
        """保存信标数据库到文件"""
        try:
            with open(beacon_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.beacon_database, f,
                          indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存信标数据库失败: {e}")

    def rssi_to_distance(self, rssi, method="improved", tx_power=None, path_loss_exponent=None, b=None, a=None):
        """
        基于RSSI计算距离 (单位: 米)
        method: "default" 使用路径损失模型，"improved" 使用线性拟合模型
        优先使用传入参数，否则使用配置文件参数。
        -281-4.51x/19.97 4.412 62.3
        """
        # 优先使用传入参数，否则用实例属性（配置文件），最后用默认值
        if tx_power is None:
            tx_power = getattr(self, 'tx_power', -53.97)
        if path_loss_exponent is None:
            path_loss_exponent = getattr(self, 'path_loss_exponent', 2.36)
        if a is None:
            a = getattr(self, 'a', -2.48)
        if b is None:
            b = getattr(self, 'b', 67.81)

        if method == "improved":
            r = (rssi + b) / a
            return max(r, 0.1)
        if method == "improved+":
            r = (rssi + 74.65) / -1.68 if rssi < -78 else (rssi + 64.8) / -6.48
            return max(r, 0.1)
        if method == "default":
            r = 3*(10**((-84.38-rssi)/(10 *2.1447)))
        # 默认模型
        if rssi == 0:
            return -1.0
        exponent = (tx_power - rssi) / (10.0 * path_loss_exponent)
        distance = math.pow(10, exponent)
        return distance

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
    
    def generate_initial_points_lhs(self, beacon_positions, last_location=None, num_points=18):
        """
        使用拉丁超立方采样生成初始点
        
        Args:
            beacon_positions: 信标位置列表 [(lat, lon), ...]
            last_location: 历史位置 (lat, lon)
            num_points: 生成的初始点数量
            
        Returns:
            初始点列表 [np.array([lat, lon]), ...]
        """
        import numpy as np
        from scipy.stats import qmc
        
        initial_points = []
        
        # 计算信标范围并扩展10%
        lats = [pos[0] for pos in beacon_positions]
        lons = [pos[1] for pos in beacon_positions]
        
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        # 扩展10%
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        lat_min_ext = lat_min - 0.1 * lat_range
        lat_max_ext = lat_max + 0.1 * lat_range
        lon_min_ext = lon_min - 0.1 * lon_range  
        lon_max_ext = lon_max + 0.1 * lon_range
        
        # 添加质心点
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)
        initial_points.append(np.array([centroid_lat, centroid_lon]))
        
        # 添加历史位置点
        if last_location is not None:
            initial_points.append(np.array(last_location))
        
        # 使用拉丁超立方采样生成剩余点
        remaining_points = num_points - len(initial_points)
        if remaining_points > 0:
            sampler = qmc.LatinHypercube(d=2)  # 2维：纬度和经度
            sample = sampler.random(n=remaining_points)
            
            # 缩放到实际范围
            for point in sample:
                lat = lat_min_ext + point[0] * (lat_max_ext - lat_min_ext)
                lon = lon_min_ext + point[1] * (lon_max_ext - lon_min_ext)
                initial_points.append(np.array([lat, lon]))
        
        return initial_points
    
    def weight(self, d, mu=6.0, sigma=3.0):
        return 0.5+0.5*math.exp(-((d - mu)**2) / (2 * sigma**2))
    
    def scipy_trilateration(self, beacon_positions, distances, last_location=None, test_flag=False, lambda_weight=0.01):
        """
        使用scipy.optimize.minimize进行梯度下降定位，详细输出调试信息
        beacon_positions: [(lat1, lon1), (lat2, lon2), ...]
        distances: [d1, d2, ...]
        """
        import numpy as np
        from scipy.optimize import minimize
        if len(beacon_positions) < 3 and STRATEGY2 != 'all':
            print("[scipy_trilateration] 信标数量不足3个，无法定位")
            return None
        step_log = []
        def error_function(point):
            lat, lon = point
            total_error = 0
            details = []
            for i, (beacon_lat, beacon_lon) in enumerate(beacon_positions):
                calculated_distance = self.haversine_distance(lat, lon, beacon_lat, beacon_lon)
                distance_diff = calculated_distance - distances[i]
                #n = self.path_loss_exponent
                #weight = abs(10 * n / (distances[i] * np.log(10)) if calculated_distance > 0 else 0)
                error = (distance_diff ** 2) * self.weight(calculated_distance)
                total_error += error
                details.append(f"  信标{i+1}: 计算距离={calculated_distance:.3f}m, 期望={distances[i]:.3f}m, 差值={distance_diff:+.3f}m, 误差={error:.6f}")
            # 记录每步
            total_error += lambda_weight * self.haversine_distance(lat, lon, last_location[0], last_location[1]) if last_location else 0
            step_log.append({
                'point': (lat, lon),
                'loss': total_error,
                'details': details.copy()
            })
            return total_error

        # 生成多个初始点
        test_flag = True
        initial_points = self.generate_initial_points_lhs(beacon_positions, last_location, num_points=10)
        print(f"[scipy_trilateration] 生成了{len(initial_points)}个初始点")
        
        best_result = None
        best_error = float('inf')
        
        # 调用scipy.optimize.minimize，增加callback输出
        def callback(xk):
            if test_flag:
                print(f"当前点: ({xk[0]:.6f}, {xk[1]:.6f})，当前loss: {error_function(xk):.6f}")
        
        # 对每个优化方法尝试所有初始点
        methods = ['L-BFGS-B', 'Powell', 'TNC', 'SLSQP']
        
        for testmethod in methods:
            if best_result is not None:
                break
                
            print(f"[scipy_trilateration] 尝试优化方法: {testmethod}")
            
            for i, initial_point in enumerate(initial_points):
                try:
                    if test_flag:
                        print(f"  [初始点 {i+1}/{len(initial_points)}] ({initial_point[0]:.6f}, {initial_point[1]:.6f})")
                    # 根据不同方法设置合适的选项
                    if testmethod == 'L-BFGS-B':
                        options = {'disp': test_flag, 'maxiter': 200}
                    elif testmethod == 'Powell':
                        options = {'disp': test_flag}
                    elif testmethod == 'TNC':
                        options = {'disp': test_flag}
                    elif testmethod == 'SLSQP':
                        options = {'disp': test_flag, 'maxiter': 200}
                    else:
                        options = {'disp': test_flag}
                    
                    res = minimize(error_function, initial_point, method=testmethod, 
                                 callback=callback if test_flag else None, 
                                 options=options)
                    
                    if res.success and res.fun < best_error:
                        best_result = res
                        best_error = res.fun
                        if test_flag:
                            print(f"    [更好的结果] 方法: {testmethod}, 误差: {res.fun:.6f}")
                            
                except Exception as e:
                    if test_flag:
                        print(f"    [初始点{i+1}优化失败] {e}")
                    continue
        
        if best_result is None:
            print("[scipy_trilateration] 所有方法和初始点优化都失败")
            return None
            
        res = best_result
        if test_flag:
            print(f"[scipy_trilateration] 多初始点优化完成，最佳误差: {best_error:.6f}")
            print(f"[scipy_trilateration] 优化结束，success={res.success}, message={res.message}")
        if not res.success:
            print(f"[scipy_trilateration] 优化失败，最后loss: {res.fun:.6f}")
            for step in step_log[-5:]:
                print(f"  [失败前步] 点: {step['point']}, loss: {step['loss']:.6f}")
                for d in step['details']:
                    print(d)
            return None

        result_lat, result_lon = res.x
        print(f"[scipy_trilateration] 最终结果: ({result_lat:.6f}, {result_lon:.6f}), loss: {res.fun:.6f}")
        # 合理性检查
        max_distance_to_beacons = 0
        for beacon_lat, beacon_lon in beacon_positions:
            dist = self.haversine_distance(result_lat, result_lon, beacon_lat, beacon_lon)
            max_distance_to_beacons = max(max_distance_to_beacons, dist)
        if max_distance_to_beacons > 1000:
            print(f"[scipy_trilateration] 结果不合理，距离信标过远: {max_distance_to_beacons:.1f}米")
            return None
        if not (-90 <= result_lat <= 90) or not (-180 <= result_lon <= 180):
            print(f"[scipy_trilateration] 结果超出地理坐标范围: ({result_lat}, {result_lon})")
            return None
        if test_flag:
            print(f"[scipy_trilateration] 优化结果: ({result_lat:.6f}, {result_lon:.6f}), 优化误差: {res.fun:.6f}")
        return [result_lat, result_lon]
        
    def weighted_centroid(self, beacon_positions, rssi_values):
        """基于RSSI权重的质心算法"""
        if not beacon_positions:
            return None

        # 将RSSI转换为权重（RSSI越高，距离越近，权重越大）
        weights = []
        for rssi in rssi_values:
            # 将负的RSSI转换为正权重
            weight = max(0, rssi + 70)  # 假设最小RSSI为-100
            #weight = math.log10(1 + math.exp(rssi + 70))
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

    def normal_trilateral(self, beacon_positions, distances):
        """
        线性三边定位算法（二维）
        beacon_positions: [(lat1, lon1), (lat2, lon2), (lat3, lon3), ...]
        distances: [d1, d2, d3, ...]
        返回: [lat, lon] 或 None
        """
        import numpy as np
        n = len(beacon_positions)
        if n < 3 or len(distances) < 3:
            return None
        # 只取前3个信标
        positions = beacon_positions[:3]
        ds = distances[:3]
        # 构造a矩阵和b矩阵
        a = np.zeros((2, 2))
        b = np.zeros((2, 1))
        for i in range(2):
            a[i][0] = 2 * (positions[i][0] - positions[2][0])
            a[i][1] = 2 * (positions[i][1] - positions[2][1])
        for i in range(2):
            b[i][0] = (
                positions[i][0] ** 2 - positions[2][0] ** 2
                + positions[i][1] ** 2 - positions[2][1] ** 2
                + ds[2] ** 2 - ds[i] ** 2
            )
        try:
            # 求解线性方程组 a * [x, y]^T = b
            result = np.linalg.solve(a, b)
            lat = result[0][0]
            lon = result[1][0]
            return [lat, lon]
        except Exception as e:
            print(f"trilateral线性解算失败: {e}")
            return None

    def calculate_terminal_location(self, bluetooth_readings, method="weighted_centroid"):
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
                #distance_method:
                distances.append(self.rssi_to_distance(rssi))
                rssi_values.append(rssi)

        # 调用共用方法处理不同信标数量的情况
        return self._process_beacons_by_count(valid_readings, beacon_positions, distances, rssi_values, method)

    def _process_beacons_by_count(self, valid_readings, beacon_positions, distances, rssi_values, method="weighted_centroid"):
        """
        根据信标数量选择不同的定位算法（共用方法）
        
        Args:
            valid_readings: 有效的蓝牙读数列表
            beacon_positions: 信标位置列表 [[lat, lon], ...]
            distances: 距离列表
            rssi_values: RSSI值列表  
            method: 定位方法名称
            
        Returns:
            dict: 定位结果
        """
        beacon_count = len(valid_readings)
        
        if beacon_count == 0:
            return {
                "status": "error",
                "message": "没有找到已知位置的信标",
                "beacon_count": 0
            }
        elif beacon_count == 1 and STRATEGY2 != 'all':
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
        elif beacon_count == 2 and STRATEGY2 != 'all':
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
            # 三个及以上信标，根据method参数选择算法
            if method == "trilateration":
                # 获取历史位置
                #last_location = None
                #if hasattr(self, 'location_history') and self.location_history and len(self.location_history) > 0:
                #    last_location = self.location_history[-1]
                #result = self.trilateration(beacon_positions, distances, last_location, test_flag=False)
                #used_method = "trilateration"
                print("请更换方法，trilateration方法已弃用 ")
            elif method == "scipy_trilateration":
                last_location = None
                if hasattr(self, 'location_history') and self.location_history and len(self.location_history) > 0:
                    last_location = self.location_history[-1]
                result = self.scipy_trilateration(beacon_positions, distances, last_location, test_flag=False)
                used_method = "scipy_trilateration"
            elif method == "weighted_centroid" or method == "weight_centroid":
                result = self.weighted_centroid(beacon_positions, rssi_values)
                used_method = "weighted_centroid"
            elif method == "normal_trilateral":
                result = self.normal_trilateral(beacon_positions, distances)
                used_method = "normal_trilateral"
            elif method == "simple_centroid":
                lat = sum(pos[0] for pos in beacon_positions) / len(beacon_positions)
                lon = sum(pos[1] for pos in beacon_positions) / len(beacon_positions)
                result = [lat, lon]
                used_method = "simple_centroid"
            else:
                # 默认三边测量
                last_location = None
                if hasattr(self, 'location_history') and self.location_history and len(self.location_history) > 0:
                    last_location = self.location_history[-1]
                result = self.scipy_trilateration(beacon_positions, distances, last_location)
                used_method = "trilateration"
                
            if result:
                accuracy = sum(distances) / len(distances)
                return {
                    "status": "success",
                    "latitude": result[0],
                    "longitude": result[1],
                    "accuracy": accuracy,
                    "beacon_count": beacon_count,
                    "method": used_method
                }
            else:
                # 主方法失败，尝试加权质心作为备选
                fallback = self.weighted_centroid(beacon_positions, rssi_values)
                if fallback:
                    return {
                        "status": "fallback",
                        "latitude": fallback[0],
                        "longitude": fallback[1],
                        "accuracy": sum(distances) / len(distances),
                        "beacon_count": beacon_count,
                        "method": "weighted_centroid_fallback"
                    }
                else:
                    return {
                        "status": "error",
                        "message": "所有定位方法都失败",
                        "beacon_count": beacon_count
                    }

class DataProcessingService:
    """数据处理服务类，负责位置数据的过滤和后处理"""
    
    def __init__(self):
        # 存储历史位置数据用于滤波
        self.position_history = []  # 存储最近的位置数据
        self.max_history_size = 3   # 保存最近5个位置用于中值滤波
        
        # EMA滤波参数
        self.ema_alpha = 0.3  # EMA平滑因子，0-1之间，越小越平滑
        self.ema_last_position = None
        
        # 卡尔曼滤波参数
        self.kf_initialized = False
        self.kf_state = None  # [lat, lon, lat_velocity, lon_velocity]
        self.kf_covariance = None
        self.process_noise = 0.01  # 过程噪声
        self.measurement_noise = 0.1  # 测量噪声
    
    def filter(self, location_result):
        """应用所有过滤器到位置结果"""
        location_result = self.filter_middle(location_result)
        location_result = self.filter_ema(location_result)
        location_result = self.filter_kf(location_result)
        # 这里可以添加更多的过滤方法
        return location_result
    
    def filter_middle(self, location_result):
        """
        二维中值（medoid）滤波：
        在 position_history（长度 <= max_history_size）中选出
        距离其它点总和最小的那一个作为输出。
        """
        # -------- 0. 基本合法性检查 --------
        if (
            not location_result
            or location_result.get("status") not in {"success", "fallback", "single_beacon"}
        ):
            return location_result

        # -------- 1. 把当前点加入历史 --------
        current_pos = {
            "latitude":  location_result["latitude"],
            "longitude": location_result["longitude"],
            "timestamp": location_result.get("timestamp", ""),
        }
        self.position_history.append(current_pos)

        # 限制窗口大小
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)

        # 若窗口未满，直接返回当前值（或根据需求返回简单平均）
        if len(self.position_history) < self.max_history_size:
            return location_result

        # -------- 2. 计算每个点到其它点的距离平方和 --------
        import math

        def sqdist(p, q):
            # 在几十米到几百米的室内场景，直接用平面欧氏就够；如需大范围可改为 haversine
            return (p["latitude"] - q["latitude"]) ** 2 + (p["longitude"] - q["longitude"]) ** 2

        best_idx, best_score = 0, float("inf")
        for i, p in enumerate(self.position_history):
            score = sum(sqdist(p, q) for j, q in enumerate(self.position_history) if i != j)
            if score < best_score:
                best_idx, best_score = i, score

        # -------- 3. 选中“最居中”的原始点 --------
        median_pos = self.position_history[best_idx]
        location_result["latitude"]  = median_pos["latitude"]
        location_result["longitude"] = median_pos["longitude"]

        return location_result

    
    def filter_ema(self, location_result):
        """指数移动平均滤波"""
        if not location_result or location_result.get("status") not in ["success", "fallback", "single_beacon"]:
            return location_result
        
        current_lat = location_result['latitude']
        current_lon = location_result['longitude']
        
        if self.ema_last_position is None:
            # 第一次使用，直接使用当前位置
            self.ema_last_position = {
                'latitude': current_lat,
                'longitude': current_lon
            }
            return location_result
        
        # 应用EMA公式: new_value = alpha * current + (1 - alpha) * previous
        filtered_lat = self.ema_alpha * current_lat + (1 - self.ema_alpha) * self.ema_last_position['latitude']
        filtered_lon = self.ema_alpha * current_lon + (1 - self.ema_alpha) * self.ema_last_position['longitude']
        
        # 更新历史位置
        self.ema_last_position = {
            'latitude': filtered_lat,
            'longitude': filtered_lon
        }
        
        # 更新结果
        location_result['latitude'] = filtered_lat
        location_result['longitude'] = filtered_lon
        
        return location_result
    
    def filter_kf(self, location_result):
        """卡尔曼滤波"""
        if not location_result or location_result.get("status") not in ["success", "fallback", "single_beacon"]:
            return location_result
        
        import numpy as np
        
        current_lat = location_result['latitude']
        current_lon = location_result['longitude']
        
        if not self.kf_initialized:
            # 初始化卡尔曼滤波器
            # 状态向量: [lat, lon, lat_velocity, lon_velocity]
            self.kf_state = np.array([current_lat, current_lon, 0.0, 0.0])
            # 初始协方差矩阵
            self.kf_covariance = np.eye(4) * 0.1
            self.kf_initialized = True
            return location_result
        
        # 状态转移矩阵 (假设单位时间步长)
        dt = 1.0  # 时间步长
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 过程噪声协方差矩阵
        Q = np.eye(4) * self.process_noise
        
        # 观测矩阵 (只观测位置，不观测速度)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 测量噪声协方差矩阵
        R = np.eye(2) * self.measurement_noise
        
        # 预测步骤
        predicted_state = F @ self.kf_state
        predicted_covariance = F @ self.kf_covariance @ F.T + Q
        
        # 更新步骤
        measurement = np.array([current_lat, current_lon])
        innovation = measurement - H @ predicted_state
        innovation_covariance = H @ predicted_covariance @ H.T + R
        
        # 卡尔曼增益
        try:
            kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            kalman_gain = predicted_covariance @ H.T @ np.linalg.pinv(innovation_covariance)
        
        # 更新状态和协方差
        self.kf_state = predicted_state + kalman_gain @ innovation
        self.kf_covariance = (np.eye(4) - kalman_gain @ H) @ predicted_covariance
        
        # 更新结果
        location_result['latitude'] = self.kf_state[0]
        location_result['longitude'] = self.kf_state[1]
        
        return location_result

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
        self.bluetooth_csv_path = "./other_data/bluetooth_position_data.csv"
        if not os.path.exists("./other_data"):
            os.makedirs("./other_data")
        if not os.path.exists(self.bluetooth_csv_path):
            # 创建CSV文件头部
            bluetooth_df = pd.DataFrame(
                columns=["id", "device_id", "mac", "rssi", "rotation", "timestamp"])
            bluetooth_df.to_csv(self.bluetooth_csv_path, index=False)

        # 初始化定位计算器
        self.location_calculator = BeaconLocationCalculator(self.config_manager)
        self.location_calculator.load_beacon_database()

        # 初始化数据处理服务
        self.data_processing_service = DataProcessingService()

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
            location_result_unfiltered = self.location_calculator.calculate_terminal_location(readings)
            location_result = self.data_processing_service.filter(location_result_unfiltered)

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
                if self.fn_location:
                    self.fn_location(location_data)

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
            parts = payload.split(';')  # 拆分为每组三元组和最后的设备号

            # 去除每组三元组中第一项的前导0
            processed_parts = []
            for part in parts[:-1]:  # 最后一项是设备号，不处理
                items = part.split(',')
                if len(items) == 3:
                    # 去除第一个字段（mac地址）前导0
                    items[0] = items[0].lstrip('0')
                    processed_parts.append(','.join(items))
                else:
                    processed_parts.append(part)  # 防止意外格式错误

            # 拼接处理后的字符串，加上设备号
            processed_payload = ';'.join(processed_parts) + ';' + parts[-1]
            #message = f"[{datetime.now().strftime('%H:%M:%S')}] 收到消息 - 主题: {topic}, 内容: {payload}"
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 收到消息  内容: {processed_payload}"
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
            location_result_unfiltered = self.location_calculator.calculate_terminal_location(readings)
            location_result = self.data_processing_service.filter(location_result_unfiltered)

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

                # 传递位置数据给GUI用于可视化（包括原始和滤波后数据）
                self.fn_location(location_data) if self.fn_location else None

                message = f"[{datetime.now().strftime('%H:%M:%S')}] 位置{self.location_id_counter}计算成功: ({location_result['latitude']:.6f}, {location_result['longitude']:.6f}), 方法: {location_result['method']}, 信标数: {location_result['beacon_count']}"
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
            self.fn_message(error_message) if self.fn_message else None

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
    
    def process_local_bluetooth_file(self,file_path="data.xlsx"):
        if not os.path.exists(file_path):
            print(f"本地蓝牙数据文件不存在：{file_path}")
            return
        try:
            df = pd.read_excel(file_path,sheet_name="bluetooth_position_data")
            grouped = df.groupby(['device_id','timestamp','id'])
            for (device_id, timestamp,id), group in grouped:
                # 组装成和 handle_bluetooth_position_data 一样的格式
                # 格式: mac,rssi,rotation;mac,rssi,rotation;...;device_id
                items = [
                    f"{row['mac']},{int(row['rssi'])},{int(row['rotation'])}"
                    for _, row in group.iterrows()
                ]
                data_str = ";".join(items) + f";{device_id}"
                try:
                    bluetooth_results = self.handle_bluetooth_position_data(data_str)
                    # 只调用一次定位计算，既保存到CSV又更新可视化
                    self.calculate_and_save_location(bluetooth_results)
                except Exception as e:
                    print(f"处理本地蓝牙数据时出错: {e}")

            print("本地蓝牙数据处理完成。")
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
          
class DataMonitorGUI:
    def __init__(self, processor: MQTTDataProcessor, config_manager: ConfigManager):
        self.processor = processor
        self.config_manager = config_manager
        self.message_queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title("蓝牙信标定位监控系统")
        self.root.geometry("1200x800")

        # 位置历史记录 - 支持多设备
        self.location_history = {}  # key为device_id
        # 新增：多方法历史
        self.location_history_method = {}  # {method: {device_id: [位置列表]}}
        # 新增：原始位置历史记录（未滤波）
        
        # 当前处理的文件名
        self.current_file_name = None


        # MQTT客户端管理
        self.mqtt_client = None
        self.mqtt_thread = None

        # 测试模式相关变量
        self.test_mode = False
        self.test_data = []
        self.test_index = 0
        
        # 显示半径相关变量
        self.show_radius_enabled = False
        self.beacon_distances = {}  # 存储当前计算的信标距离
        
        # 比较模式控制变量
        self.FILTERCOMPARE = FILTERCOMPARE   # 滤波器比较模式
        self.METHODCOMPARE = METHODCOMPARE  # 方法比较模式
        
        # 显示控制变量
        self.show_annotations = True  # 控制是否显示路径序号和方法标注

        # 绑定位置回调，确保所有位置数据都能更新历史和可视化
        self.processor.on_location(self.handle_location_data)

        self.setup_gui()

    def handle_location_data(self, location_data):
        """接收MQTTDataProcessor传来的位置数据，更新历史并刷新可视化"""
        device_id = location_data.get("device_id", "Unknown")
        if device_id not in self.location_history:
            self.location_history[device_id] = []
        self.location_history[device_id].append(location_data)
        
        # 自动刷新可视化
        if hasattr(self, "auto_update_var") and getattr(self.auto_update_var, "get", lambda: True)():
            self.update_visualization()

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

        # 新增：导入本地蓝牙数据按钮
        ttk.Button(button_row1, text="导入本地蓝牙数据", command=self.import_local_bluetooth_file).pack(side=tk.LEFT, padx=(10, 10))
        # 新增：使用本地蓝牙数据按钮
        ttk.Button(button_row1, text="使用本地蓝牙数据", command=self.use_default_local_bluetooth_file).pack(side=tk.LEFT, padx=(10, 10))
        # 新增：测试按钮
        ttk.Button(button_row1, text="测试", command=self.start_test_mode).pack(side=tk.LEFT, padx=(10, 10))
        # 新增：下一条按钮
        self.next_test_button = ttk.Button(button_row1, text="下一条", command=self.next_test_data, state="disabled")
        self.next_test_button.pack(side=tk.LEFT, padx=(10, 10))

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
        ttk.Button(button_frame, text="导入JSON", command=self.import_beacon_json).pack(
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
        # 新增：下一条按钮（可视化tab）
        self.next_test_button_viz = ttk.Button(control_frame, text="下一条", command=self.next_test_data, state="disabled")
        self.next_test_button_viz.pack(side=tk.LEFT, padx=(10, 10))
        
        # 新增：显示半径按钮
        self.show_radius_button = ttk.Button(control_frame, text="显示半径", command=self.toggle_radius_display, state="disabled")
        self.show_radius_button.pack(side=tk.LEFT, padx=(10, 10))
        
        # 新增：显示标注按钮
        self.show_annotations_button = ttk.Button(control_frame, text="隐藏标注", command=self.toggle_annotations_display)
        self.show_annotations_button.pack(side=tk.LEFT, padx=(10, 10))

        # 创建matplotlib图形
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 创建Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始化图形
        self.init_plot()

    def get_plot_title(self, base_title="蓝牙信标定位可视化"):
        """生成包含文件名的绘图标题"""
        if hasattr(self, 'current_file_name') and self.current_file_name:
            return f"{base_title} - {self.current_file_name}"
        return base_title

    def init_plot(self):
        """初始化绘图"""
        self.ax.clear()
        self.ax.set_title(self.get_plot_title())
        self.ax.set_xlabel("经度")
        self.ax.set_ylabel("纬度")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def update_visualization(self):
        """更新可视化图形"""
        if getattr(self, 'test_mode', False):
            self.update_visualization_test_mode()
            return
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
                                     alpha=0.6, linewidth=2, label=f'设备{color_index}轨迹')
                        # 方法缩写映射
                        method_abbr = {
                            'trilateration': 't',
                            'scipy_trilateration': 't',
                            'weighted_centroid': 'c',
                            'weighted_centroid_fallback': 'c',
                            'simple_centroid': 's',
                            'normal_trilateral': 'n',
                            'single_beacon': 'b'
                        }
                        # 只有在启用标注显示时才添加标注
                        if self.show_annotations:
                            for idx, (lon, lat, loc_data) in enumerate(zip(device_lons, device_lats, device_history), start=1):
                                if idx != len(device_lons):
                                    method = loc_data.get('calculation_method', 'unknown')
                                    method_short = method_abbr.get(method, 'u')
                                    self.ax.annotate(f"{idx}{method_short}", (lon, lat), xytext=(10, -10), textcoords='offset points', fontsize=10, color=color)
                    # 绘制该设备的当前位置
                    if device_history:
                        current = device_history[-1]
                        self.ax.scatter([current['longitude']], [current['latitude']],
                                        c=color, s=150, marker='o', 
                                        alpha=0.9, edgecolors='black', linewidth=2)
                        
                        # 添加设备ID标签
                        self.ax.annotate(f'设备{color_index}', 
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

            self.ax.set_title(self.get_plot_title())
            self.ax.set_xlabel("经度")
            self.ax.set_ylabel("纬度")
            self.ax.ticklabel_format(style='plain', useOffset=False, axis='both')
            self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
            self.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal', adjustable='box')  # 确保横纵比例尺相同
            self.ax.legend()

            self.canvas.draw()

        except Exception as e:
            print(f"更新可视化时出错: {e}")

    def update_visualization_test_mode(self):
        """测试模式下的可视化，支持多方法轨迹"""
        if not self.processor:
            return
        try:
            self.ax.clear()
            beacons = self.processor.location_calculator.get_all_beacons()
            if beacons:
                beacon_lons = [info['longitude'] for info in beacons.values()]
                beacon_lats = [info['latitude'] for info in beacons.values()]
                self.ax.scatter(beacon_lons, beacon_lats, c='blue', s=100, marker='^',
                                label='信标位置', alpha=0.8, edgecolors='darkblue')
                for mac, info in beacons.items():
                    self.ax.annotate(mac[-4:], (info['longitude'], info['latitude']),
                                     xytext=(5, 5), textcoords='offset points', fontsize=8)
                method_colors = {
                    # 原算法方法颜色
                    "trilateration": "red",
                    "weighted_centroid": "green", 
                    "simple_centroid": "orange",
                    "normal_trilateral": "purple",
                    "scipy_trilateration": "red",
                    # 滤波器比较模式颜色
                    "no_filter": "red",      # 无滤波器 - 红色
                    "filter1": "green",      # 1个滤波器 - 绿色  
                    "filter2": "blue",       # 2个滤波器 - 蓝色
                    "filter3": "purple"      # 3个滤波器 - 紫色
                }
                all_lons = beacon_lons.copy()
                all_lats = beacon_lats.copy()
                for method, device_dict in self.location_history_method.items():
                    color = method_colors.get(method, "gray")
                    for device_id, device_history in device_dict.items():
                        device_lons = [loc['longitude'] for loc in device_history]
                        device_lats = [loc['latitude'] for loc in device_history]
                        all_lons.extend(device_lons)
                        all_lats.extend(device_lats)
                        if len(device_lons) > 1:
                            self.ax.plot(device_lons, device_lats, color=color,
                                         alpha=0.6, linewidth=2, label=f'方法{method}轨迹')
                            # 方法缩写映射
                            method_abbr = {
                                'trilateration': 't',
                                'scipy_trilateration': 't', 
                                'weighted_centroid': 'c',
                                'weighted_centroid_fallback': 'c',
                                'simple_centroid': 's',
                                'normal_trilateral': 'n',
                                'single_beacon': 'b'
                            }
                            # 只有在启用标注显示时才添加路径编号和实际使用的方法缩写
                            if self.show_annotations:
                                for idx, (lon, lat, loc_data) in enumerate(zip(device_lons, device_lats, device_history), start=1):
                                    if idx != len(device_lons):
                                        # 使用实际计算的方法，而不是预期的方法
                                        actual_method = loc_data.get('method', method)
                                        actual_method_short = method_abbr.get(actual_method, 'u')
                                        self.ax.annotate(f"{idx}{actual_method_short}", (lon, lat), xytext=(10, -10), textcoords='offset points', fontsize=10, color=color)
                        if device_history:
                            current = device_history[-1]
                            self.ax.scatter([current['longitude']], [current['latitude']],
                                            c=color, s=150, marker='o', 
                                            alpha=0.9, edgecolors='black', linewidth=2)
                            self.ax.annotate(f'{method}', 
                                            (current['longitude'], current['latitude']),
                                            xytext=(10, 10), textcoords='offset points', 
                                            fontsize=10, fontweight='bold',
                                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
                
                # 添加半径显示功能
                if self.show_radius_enabled and hasattr(self, 'beacon_distances') and self.beacon_distances:
                    from matplotlib.patches import Circle
                    # 获取信标位置信息
                    beacons = self.processor.location_calculator.get_all_beacons()
                    if beacons:
                        for mac, distance in self.beacon_distances.items():
                            if mac in beacons:
                                beacon_info = beacons[mac]
                                beacon_lon = beacon_info['longitude']
                                beacon_lat = beacon_info['latitude']
                                
                                # 将距离从米转换为经纬度差（近似）
                                # 1度纬度 ≈ 111111米，1度经度 ≈ 111111*cos(lat)米
                                lat_diff = distance / 111111.0  # 纬度差
                                lon_diff = distance / (111111.0 * abs(math.cos(math.radians(beacon_lat))))  # 经度差
                                
                                # 创建圆形
                                circle = Circle((beacon_lon, beacon_lat), 
                                              radius=max(lat_diff, lon_diff),  # 使用较大的差值作为半径
                                              fill=False, 
                                              color='red', 
                                              alpha=0.6, 
                                              linewidth=1.5,
                                              linestyle='--')
                                self.ax.add_patch(circle)
                                
                                # 添加距离和RSSI标签
                                rssi_value = self.beacon_rssi.get(mac, 'N/A') if hasattr(self, 'beacon_rssi') else 'N/A'
                                label_text = f'{distance:.1f}m\nRSSI: {rssi_value}dBm' if rssi_value != 'N/A' else f'{distance:.1f}m'
                                self.ax.annotate(label_text, 
                                               (beacon_lon, beacon_lat),
                                               xytext=(15, -15), textcoords='offset points',
                                               fontsize=8, color='red',
                                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                if all_lons and all_lats:
                    lon_margin = (max(all_lons) - min(all_lons)) * 0.1 or 0.001
                    lat_margin = (max(all_lats) - min(all_lats)) * 0.1 or 0.001
                    self.ax.set_xlim(min(all_lons) - lon_margin,
                                     max(all_lons) + lon_margin)
                    self.ax.set_ylim(min(all_lats) - lat_margin,
                                     max(all_lats) + lat_margin)
            self.ax.set_title(self.get_plot_title("蓝牙信标定位可视化（多方法测试）"))
            self.ax.set_xlabel("经度")
            self.ax.set_ylabel("纬度")
            self.ax.ticklabel_format(style='plain', useOffset=False, axis='both')
            self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
            self.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal', adjustable='box')  # 确保横纵比例尺相同
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            print(f"测试模式下更新可视化时出错: {e}")

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

        # 新增a参数设置
        a_frame = ttk.Frame(rssi_frame)
        a_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(a_frame, text="a参数:").pack(side=tk.LEFT, padx=(0, 5))
        self.a_entry = ttk.Entry(a_frame, width=10)
        self.a_entry.pack(side=tk.LEFT, padx=(0, 10))

        # 新增b参数设置
        b_frame = ttk.Frame(rssi_frame)
        b_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(b_frame, text="b参数:").pack(side=tk.LEFT, padx=(0, 5))
        self.b_entry = ttk.Entry(b_frame, width=10)
        self.b_entry.pack(side=tk.LEFT, padx=(0, 10))

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
        self.tx_power_entry.insert(0, str(rssi_config.get("tx_power", -53.97)))
        self.path_loss_entry.delete(0, tk.END)
        self.path_loss_entry.insert(0, str(rssi_config.get("path_loss_exponent", 2.36)))
        self.a_entry.delete(0, tk.END)
        self.a_entry.insert(0, str(rssi_config.get("a", -2.48)))
        self.b_entry.delete(0, tk.END)
        self.b_entry.insert(0, str(rssi_config.get("b", 65.81)))

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
            a = float(self.a_entry.get()) if self.a_entry.get() else -2.48
            b = float(self.b_entry.get()) if self.b_entry.get() else 65.81

            if path_loss_exponent <= 0:
                messagebox.showerror("错误", "路径损失指数必须大于0")
                return

            # 保存到配置
            if "rssi_model" not in self.config_manager.config:
                self.config_manager.config["rssi_model"] = {}
            self.config_manager.config["rssi_model"]["tx_power"] = tx_power
            self.config_manager.config["rssi_model"]["path_loss_exponent"] = path_loss_exponent
            self.config_manager.config["rssi_model"]["a"] = a
            self.config_manager.config["rssi_model"]["b"] = b
            self.config_manager.save_config()

            # 更新处理器中的参数
            if self.processor:
                self.processor.location_calculator.update_rssi_model_params(tx_power, path_loss_exponent,a,b)
            
            self.update_config_info_display()
            messagebox.showinfo("成功", "RSSI模型设置已保存")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")

    def reset_rssi_settings(self):
        """恢复RSSI模型默认设置"""
        self.tx_power_entry.delete(0, tk.END)
        self.tx_power_entry.insert(0, "-53.97")
        self.path_loss_entry.delete(0, tk.END)
        self.path_loss_entry.insert(0, "2.36")
        self.a_entry.delete(0, tk.END)
        self.a_entry.insert(0, "-2.48")
        self.b_entry.delete(0, tk.END)
        self.b_entry.insert(0, "65.81")

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
            config_text += f"  a参数: {rssi_config['a']}\n"
            config_text += f"  b参数: {rssi_config['b']}\n\n"
            
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
        self.location_history_method.clear()
        self.location_history_raw.clear()
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

    def import_beacon_json(self):
        """导入JSON信标文件"""
        from tkinter import filedialog
        import json
        
        if not self.processor:
            messagebox.showerror("错误", "处理器未初始化")
            return
            
        try:
            # 打开文件选择对话框
            file_path = filedialog.askopenfilename(
                title="选择信标JSON文件",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
            if not file_path:
                return
                
            # 读取并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                beacon_data = json.load(f)
            
            # 验证数据格式
            if not isinstance(beacon_data, dict):
                messagebox.showerror("错误", "JSON文件格式错误：根对象应为字典")
                return
            
            imported_count = 0
            skipped_count = 0
            
            # 导入信标数据
            for mac_address, info in beacon_data.items():
                try:
                    if not isinstance(info, dict):
                        print(f"跳过格式错误的信标: {mac_address}")
                        skipped_count += 1
                        continue
                        
                    longitude = info.get('longitude')
                    latitude = info.get('latitude') 
                    altitude = info.get('altitude', 0.0)
                    
                    if longitude is None or latitude is None:
                        print(f"跳过缺少坐标的信标: {mac_address}")
                        skipped_count += 1
                        continue
                    
                    # 添加或更新信标
                    self.processor.location_calculator.add_beacon(
                        mac_address, longitude, latitude, altitude
                    )
                    imported_count += 1
                    
                except Exception as e:
                    print(f"导入信标 {mac_address} 失败: {e}")
                    skipped_count += 1
                    continue
            
            # 刷新列表显示
            self.refresh_beacon_list()
            
            # 显示导入结果
            if imported_count > 0:
                messagebox.showinfo(
                    "导入完成", 
                    f"成功导入 {imported_count} 个信标\n跳过 {skipped_count} 个信标"
                )
            else:
                messagebox.showwarning("导入完成", "没有成功导入任何信标")
                
        except json.JSONDecodeError:
            messagebox.showerror("错误", "JSON文件格式错误，无法解析")
        except Exception as e:
            messagebox.showerror("错误", f"导入文件失败: {e}")

    def clear_entries(self):
        """清空输入框"""
        self.mac_entry.delete(0, tk.END)
        self.lon_entry.delete(0, tk.END)
        self.lat_entry.delete(0, tk.END)
        self.alt_entry.delete(0, tk.END)

    def import_local_bluetooth_file(self):
        self.test_mode= False
        file_path = filedialog.askopenfilename(
            title="选择本地蓝牙数据Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls")]
        )
        if not file_path:
            return
        
        # 存储当前文件名（只保留文件名，不包含路径）
        import os
        self.current_file_name = os.path.basename(file_path)
        
        # 保存路径到json文件
        try:
            if not os.path.exists(DATA_PATH_JSON_PATH):
                json.dump({"path": file_path}, open(DATA_PATH_JSON_PATH, "w", encoding="utf-8"))
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 创建保存路径成功")
            with open(DATA_PATH_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"path": file_path}, f)
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 保存路径成功")
        except Exception as e:
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 保存路径失败: {e}")
        self.processor.process_local_bluetooth_file(file_path)
        self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 已导入本地蓝牙数据文件: {file_path}")
        
        # 处理完成后自动保存绘图
        self.save_path_plot()

    def use_default_local_bluetooth_file(self):
        self.test_mode = False
        """优先使用json文件中的路径，否则使用默认本地蓝牙数据文件（如data.xlsx）"""
        file_path = "data.xlsx"
        if os.path.exists(DATA_PATH_JSON_PATH):
            try:
                with open(DATA_PATH_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "path" in data and os.path.exists(data["path"]):
                        file_path = data["path"]
            except Exception as e:
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 读取路径json失败: {e}")
        
        # 存储当前文件名（只保留文件名，不包含路径）
        self.current_file_name = os.path.basename(file_path)
        
        self.processor.process_local_bluetooth_file(file_path)
        self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 已处理本地蓝牙数据文件: {file_path}")
        
        # 处理完成后自动保存绘图
        self.save_path_plot()

    def start_test_mode(self):
        import json
        import os
        file_path = "data.xlsx"
        self.test_data = []
        self.test_index = 0
        self.test_mode = True
        if os.path.exists(DATA_PATH_JSON_PATH):
            try:
                with open(DATA_PATH_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "path" in data and os.path.exists(data["path"]):
                        file_path = data["path"]
            except Exception as e:
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 读取路径json失败: {e}")
        if not os.path.exists(file_path):
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 本地蓝牙数据文件不存在: {file_path}")
            self.next_test_button.config(state="disabled")
            self.next_test_button_viz.config(state="disabled")
            self.show_radius_button.config(state="disabled")
            return
        else:
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 使用本地蓝牙数据文件: {file_path}")
        try:
            import pandas as pd
            df = pd.read_excel(file_path, sheet_name="bluetooth_position_data")
            grouped = df.groupby(['device_id', 'timestamp', 'id'])
            for key, group in grouped:
                self.test_data.append((key, group))
            if self.test_data:
                self.next_test_button.config(state="normal")
                self.next_test_button_viz.config(state="normal")
                self.show_radius_button.config(state="normal")
                self.process_test_data(0)
            else:
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 没有可用的测试数据")
                self.next_test_button.config(state="disabled")
                self.next_test_button_viz.config(state="disabled")
                self.show_radius_button.config(state="disabled")
        except Exception as e:
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 读取Excel文件失败: {e}")
            self.next_test_button.config(state="disabled")
            self.next_test_button_viz.config(state="disabled")
            self.show_radius_button.config(state="disabled")

    def process_test_data(self, index):
        #测试数据处理函数
        if 0 <= index < len(self.test_data):
            (device_id, timestamp, id_), group = self.test_data[index]
            items = [
                f"{row['mac']},{int(row['rssi'])},{int(row['rotation'])}"
                for _, row in group.iterrows()
            ]
            data_str = ";".join(items) + f";{device_id}"
            try:
                bluetooth_results = self.processor.handle_bluetooth_position_data(data_str)
                
                # 存储信标距离信息用于半径显示
                self.beacon_distances = {}
                self.beacon_rssi = {}
                for result in bluetooth_results:
                    if 'mac' in result and 'rssi' in result:
                        distance = self.processor.location_calculator.rssi_to_distance(result['rssi'])
                        self.beacon_distances[result['mac']] = distance
                        self.beacon_rssi[result['mac']] = result['rssi']
                
                # 根据比较模式选择处理逻辑
                if self.FILTERCOMPARE:
                    # ---------- 1. 计算原始定位 ----------
                    base_method = "weighted_centroid" 
                    raw_result = self.processor.location_calculator.calculate_terminal_location(
                        bluetooth_results, method=base_method
                    )

                    # ---------- 2. 依次应用 0/1/2/3 重滤波 ----------
                    service = self.processor.data_processing_service   # 便于书写
                    filtered_map = {
                        "no_filter": None,
                        "filter1": None,
                        "filter2": None,
                        "filter3": None,
                    }

                    if raw_result:
                        raw   = deepcopy(raw_result)
                        mid   = service.filter_middle(deepcopy(raw))
                        ema   = service.filter_ema(deepcopy(mid))
                        kf    = service.filter_kf(deepcopy(ema))

                        filtered_map["no_filter"] = raw
                        filtered_map["filter1"]   = mid
                        filtered_map["filter2"]   = ema
                        filtered_map["filter3"]   = kf

                    # ---------- 3. 统一保存 / 打印 ----------
                    for filter_name, location_result in filtered_map.items():

                        # 标记使用的算法名称
                        used_method = f"{base_method}_{filter_name}"
                        if location_result is not None:
                            location_result["method"] = used_method
                        else:
                            used_method += "_failed"

                        # 写入历史
                        if (
                            location_result
                            and location_result.get("status") in {"success", "fallback", "single_beacon"}
                        ):
                            dev_id = (
                                bluetooth_results[0]["device_id"]
                                if bluetooth_results and "device_id" in bluetooth_results[0]
                                else device_id
                            )
                            self.location_history_method.setdefault(filter_name, {}).setdefault(dev_id, []).append(
                                {
                                    "longitude": location_result["longitude"],
                                    "latitude": location_result["latitude"],
                                    "timestamp": location_result.get("timestamp", ""),
                                    "accuracy": location_result.get("accuracy", 0),
                                    "method": used_method,
                                    "device_id": dev_id,
                                }
                            )

                    # ---------- 4. 结果输出 ----------
                    self.message_queue.put(
                        f"[{datetime.now().strftime('%H:%M:%S')}] 已处理第{index + 1}条测试数据, "
                        f"信标数: {raw_result.get('beacon_count', 0) if raw_result else 0}"
                    )
                        
                elif self.METHODCOMPARE:
                    # 方法比较模式：使用不同算法，都应用完整滤波器
                    methods = ["weighted_centroid"]#, "scipy_trilateration", "simple_centroid", "normal_trilateral"]
                    for method in methods:
                        # 计算原始位置结果
                        location_result_unfiltered = self.processor.location_calculator.calculate_terminal_location(
                            bluetooth_results, method=method)
                        # 应用完整滤波器
                        location_result = self.processor.data_processing_service.filter(location_result_unfiltered) if location_result_unfiltered else None
                        # 设置使用的方法名
                        if location_result:
                            used_method = location_result.get("method", method)
                        else:
                            used_method = f"{method}_failed"

                        if location_result and location_result.get("status") in ["success", "fallback", "single_beacon"]:
                            dev_id = bluetooth_results[0]["device_id"] if bluetooth_results and "device_id" in bluetooth_results[0] else device_id
                            if method not in self.location_history_method:
                                self.location_history_method[method] = {}
                            if dev_id not in self.location_history_method[method]:
                                self.location_history_method[method][dev_id] = []
                            self.location_history_method[method][dev_id].append({
                                'longitude': location_result['longitude'],
                                'latitude': location_result['latitude'],
                                'timestamp': location_result.get('timestamp', ''),
                                'accuracy': location_result.get('accuracy', 0),
                                'method': used_method,
                                'device_id': dev_id
                            })
                        # 输出方法和信标数
                        self.message_queue.put(
                            f"[{datetime.now().strftime('%H:%M:%S')}] 已处理第{index+1}条测试数据, 方法: {used_method}, 信标数: {location_result.get('beacon_count', 0) if location_result else 0}"
                        )
            except Exception as e:
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 测试处理本地蓝牙数据时出错: {e}")
        else:
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 测试数据已全部处理完毕")
            self.next_test_button.config(state="disabled")
            self.next_test_button_viz.config(state="disabled")
            self.show_radius_button.config(state="disabled")
            # 保存路径图
            self.save_path_plot()

    def next_test_data(self):
        if self.test_mode and self.test_data:
            self.test_index += 1
            if self.test_index < len(self.test_data):
                self.process_test_data(self.test_index)
            else:
                self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 测试数据已全部处理完毕")
                self.next_test_button.config(state="disabled")
                self.next_test_button_viz.config(state="disabled")
                self.show_radius_button.config(state="disabled")
                # 保存路径图
                self.save_path_plot()

    def save_path_plot(self):
        """保存当前路径图为JPG格式"""
        try:
            import os
            # 确保output目录存在
            output_dir = "./output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成文件名（当前时间 + 源文件名）
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            if hasattr(self, 'current_file_name') and self.current_file_name:
                # 去掉源文件的扩展名，添加到保存的文件名中
                source_name = os.path.splitext(self.current_file_name)[0]
                filename = f"{output_dir}/path_{current_time}_{source_name}.jpg"
            else:
                filename = f"{output_dir}/path_{current_time}.jpg"
            
            # 保存图形
            self.fig.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
            
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 路径图已保存至: {filename}")
            
        except Exception as e:
            self.message_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 保存路径图失败: {e}")

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
    
    def toggle_radius_display(self):
        """切换半径显示状态"""
        self.show_radius_enabled = not self.show_radius_enabled
        if self.show_radius_enabled:
            self.show_radius_button.config(text="隐藏半径")
        else:
            self.show_radius_button.config(text="显示半径")
        # 刷新可视化
        self.update_visualization()
    
    def toggle_annotations_display(self):
        """切换标注显示状态"""
        self.show_annotations = not self.show_annotations
        if self.show_annotations:
            self.show_annotations_button.config(text="隐藏标注")
        else:
            self.show_annotations_button.config(text="显示标注")
        # 刷新可视化
        self.update_visualization()
    
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
