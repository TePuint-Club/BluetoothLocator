from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import datetime
import logging

from .config_manager import ConfigManager
from .mqtt_processor import MQTTDataProcessor


def setup_logging():
    """配置全局日志，展示所有 logger 输出到控制台。"""
    root = logging.getLogger()
    # 清空已有处理器，避免重复输出
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 捕获 warnings 到 logging
    logging.captureWarnings(True)


def run_mqtt(args):
    config = ConfigManager(args.config)
    processor = MQTTDataProcessor(
        config
    )
    processor.start_mqtt_client()


def main(argv=None):
    # 初始化日志配置
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="ble-locator-server", description="BLE Locator Server CLI")
    parser.add_argument("--config", default=None,
                        help="配置文件路径，默认读取 ./config/config.yaml 或环境变量 BLE_LOCATOR_CONFIG")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="运行 MQTT 服务端监听")
    p_run.set_defaults(func=run_mqtt)

    args = parser.parse_args(argv)
    # 无子命令/无参数时默认启动服务器
    if not hasattr(args, "func"):
        return run_mqtt(args)
    return args.func(args)


if __name__ == "__main__":
    main()
