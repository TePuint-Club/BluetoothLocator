"""
入口转发

本项目已重构为可复用的包与 CLI：
  - 包名: ble_locator_server
  - CLI: ble-locator-server

此文件仅用于兼容 `python main.py` 的运行方式，会转发到 `ble_locator_server.cli:main`。
"""

from ble_locator_server.cli import main as _cli_main


def main():
    _cli_main()


if __name__ == "__main__":  # pragma: no cover
    main()
