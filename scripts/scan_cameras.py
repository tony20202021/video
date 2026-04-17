"""
Scan local subnet for iCSee cameras.
Checks ports: 554 (RTSP), 8899 (ONVIF), 34567 (DVRIP).

Usage:
    python scripts/scan_cameras.py
    python scripts/scan_cameras.py --subnet 192.168.0.0/24
    python scripts/scan_cameras.py --subnet 192.168.1.0/24 --timeout 0.5
"""

import argparse
import ipaddress
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

CAMERA_PORTS = {
    554: "RTSP",
    8899: "ONVIF",
    34567: "DVRIP",
}

# Хотя бы один из этих портов должен быть открыт
REQUIRED_PORTS = {554, 8899}


def check_port(ip: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def scan_host(ip: str, timeout: float) -> dict | None:
    open_ports = {
        port: name
        for port, name in CAMERA_PORTS.items()
        if check_port(ip, port, timeout)
    }
    if not open_ports.keys() & REQUIRED_PORTS:
        return None
    return {"ip": ip, "ports": open_ports}


def get_local_subnet() -> str:
    """Определить подсеть текущей машины."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        # Предполагаем маску /24
        parts = local_ip.rsplit(".", 1)
        return f"{parts[0]}.0/24"
    except OSError:
        return "192.168.1.0/24"


def scan(subnet: str, timeout: float, workers: int = 100) -> list[dict]:
    network = ipaddress.IPv4Network(subnet, strict=False)
    hosts = list(network.hosts())

    print(f"Сканирование {subnet} ({len(hosts)} хостов), порты: {list(CAMERA_PORTS.keys())}")
    print(f"Timeout: {timeout}s, потоков: {workers}\n")

    found = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_host, str(ip), timeout): str(ip) for ip in hosts}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                found.append(result)
                ports_str = ", ".join(f"{p}({n})" for p, n in sorted(result["ports"].items()))
                print(f"  [+] {result['ip']:16s}  открыты: {ports_str}")
            # Прогресс каждые 50 хостов
            if completed % 50 == 0 or completed == len(hosts):
                print(f"  ... {completed}/{len(hosts)}", end="\r", flush=True)

    print()
    return sorted(found, key=lambda x: ipaddress.IPv4Address(x["ip"]))


def print_results(cameras: list[dict]) -> None:
    if not cameras:
        print("Камеры не найдены.")
        return

    print(f"\nНайдено камер: {len(cameras)}\n")
    print(f"{'IP':<18} {'RTSP URL (субпоток)'}")
    print("-" * 80)
    for cam in cameras:
        ip = cam["ip"]
        if 554 in cam["ports"]:
            rtsp = f"rtsp://{ip}:554/user=admin&password=&channel=1&stream=1.sdp?real_stream"
        else:
            rtsp = "(RTSP порт 554 не открыт)"
        print(f"{ip:<18} {rtsp}")

    print()
    print("Примечание: подставьте реальный пароль вместо пустого поля password=")
    print("Сохраните URL в файл .env (не в git):")
    for i, cam in enumerate(cameras, 1):
        ip = cam["ip"]
        print(f"  CAM_{i:02d}_URL=rtsp://{ip}:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream")


def main() -> None:
    default_subnet = get_local_subnet()

    parser = argparse.ArgumentParser(description="Поиск iCSee камер в локальной сети")
    parser.add_argument("--subnet", default=default_subnet, help=f"Подсеть (default: {default_subnet})")
    parser.add_argument("--timeout", type=float, default=0.3, help="Таймаут подключения в секундах (default: 0.3)")
    parser.add_argument("--workers", type=int, default=100, help="Количество параллельных потоков (default: 100)")
    args = parser.parse_args()

    try:
        ipaddress.IPv4Network(args.subnet, strict=False)
    except ValueError:
        print(f"Ошибка: неверный формат подсети '{args.subnet}'", file=sys.stderr)
        sys.exit(1)

    cameras = scan(args.subnet, args.timeout, args.workers)
    print_results(cameras)


if __name__ == "__main__":
    main()
