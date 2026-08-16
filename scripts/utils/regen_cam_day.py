"""Server-side: собрать ПОЛНЫЙ день cam3 из логов камеры и построить графики.

Проблема: motion_diff пишет CSV в каталог WALL-даты (images/<день>/), а рестарт прогона
ПЕРЕЗАПИСЫВАЕТ дневной CSV ('w') → часть дня теряется; плюс у ряда данных строки соседней
даты могли залететь в смежный каталог. Из-за этого штатный cam3-график показывает день не целиком.

Решение (всё на СЕРВЕРЕ, камеру только читаем по scp — без обработки на камере):
  1) тянем CSV за целевую дату И соседние (D-1, D, D+1) — строки нужной даты могут быть размазаны;
  2) фильтруем строки по ts_msk == дата, сшиваем, сортируем по времени суток;
  3) mono_s переписываем в СЕКУНДЫ СУТОК (из ts_msk) — иначе ось графика ломается на стыке прогонов
     (у каждого прогона свой mono с нуля);
  4) рендерим штатным 1_motion_diff.py --regen-from.

Usage:
    python scripts/utils/regen_cam_day.py 20260815
    python scripts/utils/regen_cam_day.py 20260815 --cam CAMERAS_3 \
        --out-cpu /path/cam3_cpu_fps.png --out-pts /path/cam3_pts_drift.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# CSV с покадровыми данными (в images/<день>/) — их сшиваем по дате из соседних каталогов.
DAY_CSVS = ("frames", "diffs", "pts", "saves")


def load_env(path: Path) -> dict:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split("#", 1)[0].strip().strip('"').strip("'")
        env[k.strip()] = v
    return env


def sod_from_ts(ts: str):
    """'20260815_230659_052268_msk' → секунды от начала суток (float) или None."""
    try:
        p = ts.split("_")
        hhmmss = p[1]
        micro = int(p[2]) if len(p) > 2 and p[2].isdigit() else 0
        return int(hhmmss[0:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6]) + micro / 1e6
    except Exception:
        return None


def stitch_csvs(in_dirs: list[Path], date: str, names=DAY_CSVS) -> dict[str, list[str]]:
    """Из каталогов in_dirs собрать строки за `date` для каждого CSV.

    Возвращает {name: [header, *rows]} — mono_s (col0) переписан в секунды суток,
    строки отсортированы по времени, дубли (по всему содержимому) убраны.
    """
    out: dict[str, list[str]] = {}
    for name in names:
        header = None
        rows: list[tuple[float, str]] = []
        seen: set[str] = set()
        for d in in_dirs:
            p = d / f"{name}.csv"
            if not p.is_file():
                continue
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
            for ln in lines[1:]:
                cols = ln.split(",")
                if len(cols) < 2 or cols[1][:8] != date:
                    continue
                sod = sod_from_ts(cols[1])
                if sod is None:
                    continue
                cols[0] = f"{sod:.4f}"          # mono_s → секунды суток (единая ось)
                row = ",".join(cols)
                key = ",".join(cols[1:])         # дубль = та же строка без mono
                if key in seen:
                    continue
                seen.add(key)
                rows.append((sod, row))
        if header is None:
            continue
        rows.sort(key=lambda r: r[0])
        out[name] = [header] + [r[1] for r in rows]
    return out


def _scp(user: str, host: str, remote: str, dest: Path) -> bool:
    cmd = ["scp", "-c", "aes128-gcm@openssh.com", "-o", "ConnectTimeout=10",
           "-o", "BatchMode=yes", f"{user}@{host}:{remote}", str(dest)]
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=180).returncode == 0
    except Exception:
        return False


def _neighbors(date: str) -> list[str]:
    d = datetime.strptime(date, "%Y%m%d")
    return [(d + timedelta(days=off)).strftime("%Y%m%d") for off in (-1, 0, 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Склейка полного дня cam3 из логов камеры (на сервере)")
    ap.add_argument("date", help="целевая дата YYYYMMDD")
    ap.add_argument("--cam", default="CAMERAS_3", help="метка камеры в .env (TS_<cam>*)")
    ap.add_argument("--out-cpu", default=None, help="куда скопировать charts.png (cpu/fps)")
    ap.add_argument("--out-pts", default=None, help="куда скопировать pts_chart.png")
    ap.add_argument("--keep", action="store_true", help="не удалять временный каталог")
    args = ap.parse_args()

    date = args.date
    env = load_env(REPO / ".env")
    host = env.get(f"TS_{args.cam}", "")
    user = env.get(f"TS_{args.cam}_USER", "")
    repo = env.get(f"TS_{args.cam}_REPO", "").replace("\\", "/")
    if not (host and user and repo):
        print(f"[!] нет TS_{args.cam}/_USER/_REPO в .env", file=sys.stderr)
        return 1
    base = f"{repo}/.output/pipeline/1_motion_diff"

    tmp = Path(tempfile.mkdtemp(prefix=f"camday_{date}_"))
    dates = _neighbors(date)
    # 1) тянем покадровые CSV за D-1,D,D+1 (best-effort)
    for d in dates:
        dd = tmp / d
        dd.mkdir(parents=True, exist_ok=True)
        for f in DAY_CSVS:
            _scp(user, host, f"{base}/images/{d}/{f}.csv", dd / f"{f}.csv")
    # cpu.csv тоже сшиваем из соседей (wall-время, но и рестарт-перезапись возможна)
    for d in dates:
        _scp(user, host, f"{base}/meta/{d}/cpu.csv", tmp / d / "cpu.csv")
    # run_params.json — из целевой даты (пороги для графика)
    _scp(user, host, f"{base}/meta/{date}/run_params.json", tmp / "run_params.json")

    # 2-3) сшить покадровые + cpu, переписать mono → секунды суток
    stitched = tmp / "stitched"
    stitched.mkdir()
    parts = stitch_csvs([tmp / d for d in dates], date, names=DAY_CSVS + ("cpu",))
    for name, lines in parts.items():
        (stitched / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rp = tmp / "run_params.json"
    if rp.is_file():
        (stitched / "run_params.json").write_text(rp.read_text(encoding="utf-8"), encoding="utf-8")

    if not (stitched / "frames.csv").is_file() or not (stitched / "saves.csv").is_file():
        print(f"[!] нет данных за {date} (frames/saves пусты) — камера/каталоги?", file=sys.stderr)
        if not args.keep:
            __import__("shutil").rmtree(tmp, ignore_errors=True)
        return 2

    # 4) штатный рендер
    r = subprocess.run([sys.executable, str(REPO / "scripts/pipeline/1_motion_diff.py"),
                        "--regen-from", str(stitched)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = r.returncode == 0
    charts = stitched / "charts.png"
    pts = stitched / "pts_chart.png"
    if args.out_cpu and charts.is_file():
        __import__("shutil").copy(charts, args.out_cpu)
    if args.out_pts and pts.is_file():
        __import__("shutil").copy(pts, args.out_pts)

    n_frames = max(0, len((stitched / "frames.csv").read_text(encoding="utf-8").splitlines()) - 1)
    n_diffs = max(0, len((stitched / "diffs.csv").read_text(encoding="utf-8").splitlines()) - 1) \
        if (stitched / "diffs.csv").is_file() else 0
    print(f"{date}: сшито frames={n_frames} diffs={n_diffs}  charts={'ok' if charts.is_file() else '—'}  "
          f"pts={'ok' if pts.is_file() else '—'}")
    print(f"stitched: {stitched}")
    if not args.keep and (args.out_cpu or args.out_pts):
        __import__("shutil").rmtree(tmp, ignore_errors=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
