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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

ip = "60.205.184.72"
port = 1883

class BeaconLocationCalculator:
    """基于RSSI的蓝牙信标定位算法"""

    def __init__(self):
        # 蓝牙信标位置数据库 (MAC地址 -> 位置信息)
        self.beacon_database = {}
        # RSSI-距离模型参数
        self.tx_power = -59  # 1米处的RSSI值 (dBm)
        self.path_loss_exponent = 2.0  # 路径损失指数
        # 定位历史记录
        self.location_history = []
        self.location_csv_path = "terminal_locations.csv"
        self.init_location_csv()

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
    def __init__(self):
        self.bluetooth_data = []
        self.bluetooth_id_counter = 0
        self.location_id_counter = 0
        self.lock = threading.Lock()

        # 确保CSV文件存在
        self.bluetooth_csv_path = "bluetooth_position_data.csv"

        # 初始化定位计算器
        self.location_calculator = BeaconLocationCalculator()
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
            client.subscribe("/device/blueTooth/station/+")

            message = f"[{datetime.now().strftime('%H:%M:%S')}] 已订阅主题"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 已订阅主题")
            self.fn_message(message) if self.fn_message else None
        else:
            message = f"[{datetime.now().strftime('%H:%M:%S')}] 连接失败，返回码: {rc}"
            print(message)
            self.fn_message(message) if self.fn_message else None

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')

            with self.lock:
                if topic.startswith("/device/blueTooth/station/"):
                    # 处理蓝牙数据
                    bluetooth_results = self.handle_bluetooth_position_data(
                        payload)

                    # 为每个蓝牙数据项添加ID
                    for result in bluetooth_results:
                        result["id"] = self.bluetooth_id_counter
                        self.bluetooth_data.append(result)

                    self.bluetooth_id_counter += 1

                    # 保存到CSV
                    self.save_bluetooth_data_to_csv(bluetooth_results)

                    # 计算终端位置
                    self.calculate_and_save_location(bluetooth_results)

                    message = f"[{datetime.now().strftime('%H:%M:%S')}] 蓝牙数据已处理，当前ID: {self.bluetooth_id_counter-1}"
                    print(message)
                    self.fn_message(message) if self.fn_message else None

        except Exception as e:
            error_message = f"[{datetime.now().strftime('%H:%M:%S')}] 处理消息时出错: {str(e)}"
            print(error_message)
            self.fn_message(message) if self.fn_message else None

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
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message

        try:
            client.connect(ip, port, 60)
            client.loop_forever()
        except Exception as e:
            error_message = f"[{datetime.now().strftime('%H:%M:%S')}] MQTT连接错误: {str(e)}"
            self.fn_message(error_message) if self.fn_message else None


class DataMonitorGUI:
    def __init__(self, processor: MQTTDataProcessor):
        self.processor = processor
        self.message_queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title("蓝牙信标定位监控系统")
        self.root.geometry("1200x800")

        # 位置历史记录
        self.location_history = []

        self.setup_gui()

    def setup_gui(self):
        # 创建主选项卡控件
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建各个选项卡
        self.create_monitor_tab()
        self.create_beacon_management_tab()
        self.create_visualization_tab()

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

        # 消息日志
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=20, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(
            button_frame, text="启动MQTT监听", command=self.start_mqtt)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = ttk.Button(
            button_frame, text="清空日志", command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT)

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

                # 如果有位置历史记录，绘制终端轨迹
                if self.location_history:
                    # 绘制历史轨迹
                    history_lons = [loc['longitude']
                                    for loc in self.location_history]
                    history_lats = [loc['latitude']
                                    for loc in self.location_history]

                    if len(history_lons) > 1:
                        self.ax.plot(history_lons, history_lats,
                                     'r-', alpha=0.6, linewidth=1, label='移动轨迹')

                    # 绘制当前位置
                    if self.location_history:
                        current = self.location_history[-1]
                        self.ax.scatter([current['longitude']], [current['latitude']],
                                        c='red', s=150, marker='o', label='当前位置',
                                        alpha=0.9, edgecolors='darkred')

                # 设置合适的显示范围
                all_lons = beacon_lons + [loc['longitude']
                                          for loc in self.location_history]
                all_lats = beacon_lats + [loc['latitude']
                                          for loc in self.location_history]

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

    def add_location_to_history(self, location_data):
        """添加位置数据到历史记录"""
        self.location_history.append({
            'longitude': location_data['longitude'],
            'latitude': location_data['latitude'],
            'timestamp': location_data['timestamp'],
            'accuracy': location_data['accuracy'],
            'method': location_data['calculation_method']
        })

        # 限制历史记录数量（保留最近100个位置）
        if len(self.location_history) > 100:
            self.location_history.pop(0)

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

    def start_mqtt(self):
        if self.processor:
            self.start_button.config(state="disabled")
            mqtt_thread = threading.Thread(
                target=self.processor.start_mqtt_client, daemon=True)
            mqtt_thread.start()

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

        self.bluetooth_label.config(text=f"蓝牙数据处理数量: {bluetooth_count}")
        self.location_label.config(text=f"位置计算数量: {location_count}")

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
    assert ip != "*#*#not_a_real_ip#*#*", "请设置正确的MQTT服务器IP地址"
    
    # 创建数据处理器并传入GUI引用
    processor = MQTTDataProcessor()
    # 创建GUI
    gui = DataMonitorGUI(processor)  # 先创建GUI
    processor.on_location(gui.add_location_to_history)  # 传递位置更新函数
    processor.on_gui_message(gui.message_queue.put)  # 传递日志更新函数

    # 运行GUI
    gui.run()


if __name__ == "__main__":
    main()
