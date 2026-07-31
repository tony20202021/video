#!/usr/bin/env python3
"""
Пост-регенератор графиков из сохранённых meta-файлов прогона.

Штатные графики (charts.png / cpu_chart.png) строятся ВО ВРЕМЯ прогона из памяти
(camera_run.save_charts, *_save_cpu_chart в pipeline-скриптах). Этот скрипт строит
их ИЗ УЖЕ СОХРАНЁННЫХ файлов — например, для завершённого прогона или скопированного
с другой машины (бенчмарк 1_motion_diff_3cam).

  python scripts/utils/plot_meta_charts.py <meta_dir>
  python scripts/utils/plot_meta_charts.py --cpu cpu.csv --log run.log --out DIR

Строит в <meta_dir> (или --out):
  meta_charts.png — 2–3 панели с общей осью времени:
    • ЦПУ%/утилизация/частота  (из cpu.csv, стиль camera_run.draw_cpu_on_ax)
    • интервалы между сохранёнными кадрами (из run.log, 'Готово. Время: N с.')
    • реальный fps захвата (из frames.csv, если есть) — по камерам, бин 2 с

ВАЖНО: 'Готово. Время: N с.' в motion_diff — это ВРЕМЯ С ПРОШЛОГО СОХРАНЁННОГО КАДРА
(_now - last_save_time), т.е. «сколько было тихо до этого движения», а НЕ время обработки
одного кадра. Камера читается ~12 кадр/с непрерывно (дешёвый diff, низкий ЦПУ), а кадр
сохраняется лишь при diff>порога или раз в HEARTBEAT (пульс).
Реальное время обработки кадра (gray+diff) — в той же строке как 'счёт/кадр: min/avg/max мс'
(скользящее окно; как у yolo/classify/identify); статус показывает именно его.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─── Парсеры (чистые функции — тестируемы без matplotlib/файлов) ────────────────

def load_cpu_csv(path: Path) -> list:
    """cpu.csv → cpu_log: [mono_s(float), ts_msk(str), cpu%(float), freq_pdh, freq_step, util].

    Порядок колонок совпадает с тем, что ждёт camera_run.draw_cpu_on_ax
    (r[0]=t, r[2]=cpu, r[3/4]=freq, r[5]=utility)."""
    rows: list = []
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in lines[1:]:                       # пропускаем заголовок
        p = ln.split(",")
        if len(p) < 3:
            continue
        try:
            mono = float(p[0])
            cpu = float(p[2])
        except ValueError:
            continue
        def _f(i: int) -> float:
            try:
                return float(p[i]) if len(p) > i and p[i] != "" else 0.0
            except ValueError:
                return 0.0
        rows.append([mono, p[1], cpu, _f(3), _f(4), _f(5)])
    return rows


def ts_msk_to_sec(ts: str) -> float:
    """'20260718_070322_928715_msk' → секунды от начала суток (float)."""
    parts = ts.split("_")
    if len(parts) < 2 or len(parts[1]) < 6:
        return 0.0
    hh, mm, ss = parts[1][0:2], parts[1][2:4], parts[1][4:6]
    frac = 0.0
    if len(parts) >= 3 and parts[2].isdigit():
        frac = float("0." + parts[2])
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + frac


_DUR_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\b.*?Готово\.\s*Время:\s*([\d.]+)\s*с")


def parse_run_log_durations(text: str) -> list:
    """run.log → [(t_sec_of_day(float), duration_s(float)), ...].

    Дедуп по (время, длительность): каждое событие пишется по одной строке на камеру
    (3 камеры → 3 одинаковые строки). Порядок сохраняется."""
    out: list = []
    seen: set = set()
    for ln in text.splitlines():
        m = _DUR_RE.match(ln)
        if not m:
            continue
        h, mi, s, dur = m.groups()
        t_sec = int(h) * 3600 + int(mi) * 60 + int(s)
        key = (t_sec, dur)
        if key in seen:
            continue
        seen.add(key)
        out.append((float(t_sec), float(dur)))
    return out


def _unwrap_midnight(t_secs: list) -> list:
    """Если прогон пересёк полночь (t падает) — прибавляем 86400 к последующим."""
    out, add, prev = [], 0.0, None
    for t in t_secs:
        if prev is not None and t + add < prev - 1:
            add += 86400
        out.append(t + add)
        prev = t + add
    return out


def load_frames_csv(path: Path) -> list:
    """frames.csv → [(mono_s(float), url_id(str)), ...] по успешно прочитанным кадрам (ok=1).

    Колонки: mono_s, ts_msk, url_id, ok, plausible, event."""
    out: list = []
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in lines[1:]:
        p = ln.split(",")
        if len(p) < 4:
            continue
        try:
            mono = float(p[0])
        except ValueError:
            continue
        if p[3].strip() not in ("1", ""):     # ok=0 — неудачный кадр, не считаем
            continue
        out.append((mono, p[2]))
    return out


def compute_fps_series(frames: list, bin_s: float = 2.0) -> dict:
    """[(mono_s, url_id)] → {url_id: {'t': [центры бинов, с], 'fps': [...], 'avg': X}}.
    fps в бине = число кадров / bin_s (один проход)."""
    by_cam: dict = {}
    for mono, cam in frames:
        by_cam.setdefault(cam, []).append(mono)
    series: dict = {}
    for cam, ts in by_cam.items():
        if len(ts) < 2:
            continue
        ts.sort()
        t0, t1 = ts[0], ts[-1]
        buckets: dict = {}
        for t in ts:
            b = int((t - t0) / bin_s)
            buckets[b] = buckets.get(b, 0) + 1
        keys = sorted(buckets)
        series[cam] = {
            "t":   [t0 + (b + 0.5) * bin_s for b in keys],
            "fps": [buckets[b] / bin_s for b in keys],
            "avg": len(ts) / (t1 - t0) if t1 > t0 else 0.0,
        }
    return series


def _short_cam(url_id: str) -> str:
    """'CAM_01_9_D_URL' → '01_9_D'."""
    s = url_id
    if s.upper().startswith("CAM_"):
        s = s[4:]
    if s.upper().endswith("_URL"):
        s = s[:-4]
    return s


# ─── Построение ────────────────────────────────────────────────────────────────

def build_charts(cpu_log: list, durations: list, out_path: Path,
                 frames: "list | None" = None, heartbeat_s: float = 599.0,
                 fps_bin: float = 5.0) -> dict:
    """Фигура из 2–3 панелей (ЦПУ + интервалы сохранений + опц. fps захвата),
    общая ось «минуты от старта». Возвращает словарь со статистикой."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.utils.camera_run import draw_cpu_on_ax

    # Общее начало отсчёта (мин): mono_s из cpu.csv; run.log выравниваем по стенным часам.
    run_start = None
    if cpu_log:
        run_start = ts_msk_to_sec(cpu_log[0][1]) - float(cpu_log[0][0])

    fps_series = compute_fps_series(frames, fps_bin) if frames else {}
    has_fps = bool(fps_series)
    n = 2 + (1 if has_fps else 0)
    ratios = [3, 2] + ([2] if has_fps else [])
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.7 * n), sharex=True,
                             squeeze=False, gridspec_kw={"height_ratios": ratios})
    axcol = [a[0] for a in axes]
    ax_cpu, ax_dur = axcol[0], axcol[1]
    ax_fps = axcol[2] if has_fps else None

    # ── ЦПУ (mono_s → минуты) ──
    stats: dict = {"cpu_samples": len(cpu_log), "events": len(durations)}
    if cpu_log:
        cpu_min = [[r[0] / 60.0, r[1], r[2], r[3], r[4], r[5]] for r in cpu_log]
        draw_cpu_on_ax(ax_cpu, cpu_min, title="ЦПУ (из cpu.csv)", show_mean=True)
        cvals = [r[2] for r in cpu_log]
        stats["cpu_avg"] = round(sum(cvals) / len(cvals), 1)
        stats["cpu_max"] = round(max(cvals), 1)

    # ── Длительности (run.log) ──
    if durations:
        t_raw = _unwrap_midnight([t for t, _ in durations])
        dur = [d for _, d in durations]
        if run_start is not None:
            x = [(t - run_start) / 60.0 for t in t_raw]
        else:                                   # нет cpu.csv — от первого события
            base = t_raw[0]
            x = [(t - base) / 60.0 for t in t_raw]

        work_x = [x[i] for i in range(len(dur)) if dur[i] < heartbeat_s]
        work_d = [dur[i] for i in range(len(dur)) if dur[i] < heartbeat_s]
        hb_x = [x[i] for i in range(len(dur)) if dur[i] >= heartbeat_s]
        hb_d = [dur[i] for i in range(len(dur)) if dur[i] >= heartbeat_s]

        ax_dur.vlines(work_x, 0.1, work_d, color="#2255cc", linewidth=1.0, alpha=0.5)
        ax_dur.scatter(work_x, work_d, s=14, color="#2255cc", zorder=5,
                       label=f"движение ({len(work_d)})")
        if hb_x:
            ax_dur.scatter(hb_x, hb_d, s=26, color="#999999", marker="s", zorder=5,
                           label=f"пульс/heartbeat ({len(hb_d)})")
        if work_d:
            avg = sum(work_d) / len(work_d)
            ax_dur.axhline(avg, color="orange", linestyle="--", linewidth=0.8)
            ax_dur.text(x[0] if x else 0, avg * 1.1,
                        f"средний интервал между движениями {avg:.1f} с",
                        fontsize=8, color="orange")
            stats["gap_avg_work"] = round(avg, 1)
            stats["gap_min_work"] = round(min(work_d), 1)
            stats["gap_max_work"] = round(max(work_d), 1)
        ax_dur.set_yscale("log")
        ax_dur.set_ylabel("интервал с прошлого сохранения, с (log)")
        ax_dur.set_title("Интервалы между сохранёнными кадрами — время «тишины» до движения/пульса "
                         "(run.log 'Готово. Время', НЕ время обработки кадра)")
        ax_dur.legend(loc="upper right", fontsize=8)
        ax_dur.grid(True, which="both", linestyle="--", alpha=0.3)

    # ── Реальный fps захвата (frames.csv) ──
    if ax_fps is not None:
        fps_avgs = {}
        for cam in sorted(fps_series):
            s = fps_series[cam]
            xm = [t / 60.0 for t in s["t"]]
            ax_fps.plot(xm, s["fps"], linewidth=1.0, marker=".", markersize=4,
                        alpha=0.8, label=f"{_short_cam(cam)}  avg {s['avg']:.1f} fps")
            ax_fps.axhline(s["avg"], color="orange", linestyle="--", linewidth=0.7, alpha=0.6)
            fps_avgs[_short_cam(cam)] = round(s["avg"], 1)
        ax_fps.set_ylim(bottom=0)
        ax_fps.set_ylabel("захват, кадр/с")
        ax_fps.set_title(f"Реальный fps захвата (frames.csv, ok=1, бин {fps_bin:g} с); "
                         f"кратковременные всплески > номинала — бурсты буфера после стойлов/реконнектов")
        ax_fps.legend(loc="upper right", fontsize=8)
        ax_fps.grid(True, linestyle="--", alpha=0.3)
        stats["fps_avg"] = fps_avgs

    axcol[-1].set_xlabel("время от старта, мин")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110)
    plt.close()
    stats["out"] = str(out_path)
    return stats


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta_dir", nargs="?", type=Path,
                    help="Каталог meta/<дата> с cpu.csv и run.log")
    ap.add_argument("--cpu", type=Path, help="Путь к cpu.csv (если не meta_dir)")
    ap.add_argument("--log", type=Path, help="Путь к run.log (если не meta_dir)")
    ap.add_argument("--frames", type=Path, help="Путь к frames.csv (для fps; иначе meta_dir/frames.csv)")
    ap.add_argument("--fps-bin", type=float, default=5.0, metavar="SEC",
                    help="Окно бина fps в секундах (default 5.0; меньше — виднее бурсты)")
    ap.add_argument("--out", type=Path, help="Каталог/файл вывода (default: рядом с входом)")
    args = ap.parse_args()

    cpu_path = args.cpu or (args.meta_dir / "cpu.csv" if args.meta_dir else None)
    log_path = args.log or (args.meta_dir / "run.log" if args.meta_dir else None)
    frames_path = args.frames or (args.meta_dir / "frames.csv" if args.meta_dir else None)
    if not cpu_path and not log_path:
        ap.error("укажите meta_dir или --cpu/--log")

    cpu_log = load_cpu_csv(cpu_path) if cpu_path and Path(cpu_path).exists() else []
    durations = (parse_run_log_durations(
        Path(log_path).read_text(encoding="utf-8", errors="replace"))
        if log_path and Path(log_path).exists() else [])
    frames = (load_frames_csv(frames_path)
              if frames_path and Path(frames_path).exists() else [])

    if not cpu_log and not durations:
        print("нет данных: cpu.csv и run.log пусты/отсутствуют", file=sys.stderr)
        return 1

    base_dir = args.out or (args.meta_dir if args.meta_dir
                            else (cpu_path or log_path).parent)
    if base_dir.suffix == ".png":
        out_path = base_dir
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / "meta_charts.png"

    stats = build_charts(cpu_log, durations, out_path, frames=frames, fps_bin=args.fps_bin)
    print(f"cpu.csv: {stats.get('cpu_samples', 0)} замеров"
          f" (avg {stats.get('cpu_avg', '—')}% / max {stats.get('cpu_max', '—')}%)")
    print(f"run.log: {stats.get('events', 0)} сохранений"
          f" (интервал между движениями avg {stats.get('gap_avg_work', '—')} с,"
          f" min {stats.get('gap_min_work', '—')} / max {stats.get('gap_max_work', '—')})")
    if stats.get("fps_avg"):
        print(f"frames.csv: {len(frames)} кадров, реальный fps захвата {stats['fps_avg']}")
    print(f"→ {stats['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
