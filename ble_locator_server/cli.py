from __future__ import annotations

import argparse
import signal
import sys
import threading
from datetime import datetime

from .config_manager import ConfigManager
from .mqtt_processor import MQTTDataProcessor

def run_mqtt(args):
    config = ConfigManager(args.config)
    processor = MQTTDataProcessor(
        config
    )

    t = threading.Thread(target=processor.start_mqtt_client, daemon=True)
    t.start()

    # graceful shutdown
    def handle_sigint(sig, frame):
        processor.stop_mqtt_client()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    t.join()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ble-locator-server", description="BLE Locator Server CLI")
    parser.add_argument("--config", default=None, help="配置文件路径，默认读取 ./config/config.yaml 或环境变量 BLE_LOCATOR_CONFIG")
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
