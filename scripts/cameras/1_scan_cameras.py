"""
Scan local subnet for iCSee cameras.
Checks ports: 554 (RTSP), 8899 (ONVIF), 34567 (DVRIP).

Usage:
    python scripts/cameras/1_scan_cameras.py
    python scripts/cameras/1_scan_cameras.py --subnet 192.168.0.0/24
    python scripts/cameras/1_scan_cameras.py --subnet 192.168.1.0/24 --timeout 0.5
    python scripts/cameras/1_scan_cameras.py --output .output/1_scan_cameras
"""

import argparse
import ipaddress
import json
import platform
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

MSK = timezone(timedelta(hours=3))


def _now_msk_dir() -> str:
    return datetime.now(MSK).strftime("%Y%m%d_%H%M%S_msk")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "cameras" / "1_scan_cameras"

CAMERA_PORTS = {
    554: "RTSP",
    8899: "ONVIF",
    34567: "DVRIP",
}

# Хотя бы один из этих портов должен быть открыт
REQUIRED_PORTS = {554, 8899}


def get_mac(ip: str) -> str | None:
    """Читает MAC из ARP-кэша для заданного IP (Windows и Linux)."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["arp", "-a", ip], capture_output=True, text=True, timeout=3).stdout
            for line in out.splitlines():
                if ip in line:
                    for part in line.split():
                        if len(part) == 17 and part.count("-") == 5:
                            return part.lower()
        else:
            out = subprocess.run(["arp", "-n", ip], capture_output=True, text=True, timeout=3).stdout
            for line in out.splitlines():
                if ip in line:
                    parts = line.split()
                    if len(parts) >= 3 and len(parts[2]) == 17:
                        return parts[2].lower()
    except Exception:
        pass
    return None


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
    found = sorted(found, key=lambda x: ipaddress.IPv4Address(x["ip"]))

    # Обогащаем MAC-адресами из ARP-кэша
    for cam in found:
        cam["mac"] = get_mac(cam["ip"])

    return found


def build_report(cameras: list[dict], subnet: str, timeout: float) -> dict:
    mac_groups: dict[str, list[dict]] = {}
    no_mac: list[dict] = []
    for cam in cameras:
        mac = cam.get("mac")
        if mac:
            mac_groups.setdefault(mac, []).append(cam)
        else:
            no_mac.append(cam)

    devices = []
    for mac, group in mac_groups.items():
        devices.append({
            "mac": mac,
            "ips": [c["ip"] for c in group],
            "ports": group[0]["ports"],
            "is_duplicate": len(group) > 1,
            "suggested_url": f"rtsp://{group[0]['ip']}:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream",
        })
    for cam in no_mac:
        devices.append({
            "mac": None,
            "ips": [cam["ip"]],
            "ports": cam["ports"],
            "is_duplicate": False,
            "suggested_url": f"rtsp://{cam['ip']}:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream",
        })

    return {
        "generated_at_msk": datetime.now(MSK).isoformat(),
        "subnet": subnet,
        "timeout_sec": timeout,
        "total_ips_found": len(cameras),
        "unique_devices": len(devices),
        "devices": devices,
    }


def print_results(cameras: list[dict]) -> None:
    if not cameras:
        print("Камеры не найдены.")
        return

    mac_groups: dict[str, list[dict]] = {}
    no_mac: list[dict] = []
    for cam in cameras:
        mac = cam.get("mac")
        if mac:
            mac_groups.setdefault(mac, []).append(cam)
        else:
            no_mac.append(cam)

    unique_devices = len(mac_groups) + len(no_mac)
    print(f"\nНайдено IP с открытыми портами: {len(cameras)}")
    print(f"Уникальных устройств (по MAC): {unique_devices}\n")

    print(f"{'IP':<18} {'MAC':<20} {'Примечание'}")
    print("-" * 80)
    for mac, group in mac_groups.items():
        note = f"ДУБЛЬ ({len(group)} IP, одно устройство)" if len(group) > 1 else ""
        for cam in group:
            print(f"  {cam['ip']:<16} {mac:<20} {note}")
            note = ""
    for cam in no_mac:
        print(f"  {cam['ip']:<16} {'(MAC не получен)':<20}")

    print()
    print("Примечание: подставьте реальный пароль вместо пустого поля password=")
    print("Сохраните URL уникальных устройств в файл .env (не в git):")
    idx = 1
    for mac, group in mac_groups.items():
        ip = group[0]["ip"]
        print(f"  CAM_{idx:02d}_URL=rtsp://{ip}:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream")
        idx += 1
    for cam in no_mac:
        ip = cam["ip"]
        print(f"  CAM_{idx:02d}_URL=rtsp://{ip}:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream")
        idx += 1


def main() -> None:
    default_subnet = get_local_subnet()

    parser = argparse.ArgumentParser(description="Поиск iCSee камер в локальной сети")
    parser.add_argument("--subnet", default=default_subnet, help=f"Подсеть (default: {default_subnet})")
    parser.add_argument("--timeout", type=float, default=0.3, help="Таймаут подключения в секундах (default: 0.3)")
    parser.add_argument("--workers", type=int, default=100, help="Количество параллельных потоков (default: 100)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Каталог для сохранения результатов")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результаты на диск")
    args = parser.parse_args()

    try:
        ipaddress.IPv4Network(args.subnet, strict=False)
    except ValueError:
        print(f"Ошибка: неверный формат подсети '{args.subnet}'", file=sys.stderr)
        sys.exit(1)

    cameras = scan(args.subnet, args.timeout, args.workers)
    print_results(cameras)

    if not args.no_save:
        run_id = _now_msk_dir()
        run_dir = args.output / f"scan_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        report = build_report(cameras, args.subnet, args.timeout)
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    main()
