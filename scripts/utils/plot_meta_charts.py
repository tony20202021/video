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
  meta_charts.png — 2 панели с общей осью времени:
    • ЦПУ%/утилизация/частота  (из cpu.csv, стиль camera_run.draw_cpu_on_ax)
    • длительности операций    (из run.log, строки 'Готово. Время: N с.')
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


# ─── Построение ────────────────────────────────────────────────────────────────

def build_charts(cpu_log: list, durations: list, out_path: Path,
                 heartbeat_s: float = 599.0) -> dict:
    """2-панельная фигура (CPU + длительности), общая ось «минуты от старта».
    Возвращает словарь со статистикой."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.utils.camera_run import draw_cpu_on_ax

    # Общее начало отсчёта (мин): mono_s из cpu.csv; run.log выравниваем по стенным часам.
    run_start = None
    if cpu_log:
        run_start = ts_msk_to_sec(cpu_log[0][1]) - float(cpu_log[0][0])

    fig, (ax_cpu, ax_dur) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})

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
                       label=f"цикл движения ({len(work_d)})")
        if hb_x:
            ax_dur.scatter(hb_x, hb_d, s=26, color="#999999", marker="s", zorder=5,
                           label=f"пульс/heartbeat ({len(hb_d)})")
        if work_d:
            avg = sum(work_d) / len(work_d)
            ax_dur.axhline(avg, color="orange", linestyle="--", linewidth=0.8)
            ax_dur.text(x[0] if x else 0, avg * 1.1,
                        f"среднее (без пульса) {avg:.1f} с",
                        fontsize=8, color="orange")
            stats["dur_avg_work"] = round(avg, 1)
            stats["dur_min_work"] = round(min(work_d), 1)
            stats["dur_max_work"] = round(max(work_d), 1)
        ax_dur.set_yscale("log")
        ax_dur.set_ylabel("длительность, с (log)")
        ax_dur.set_title("Длительности операций (run.log: 'Готово. Время')")
        ax_dur.legend(loc="upper right", fontsize=8)
        ax_dur.grid(True, which="both", linestyle="--", alpha=0.3)

    ax_dur.set_xlabel("время от старта, мин")
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
    ap.add_argument("--out", type=Path, help="Каталог/файл вывода (default: рядом с входом)")
    args = ap.parse_args()

    cpu_path = args.cpu or (args.meta_dir / "cpu.csv" if args.meta_dir else None)
    log_path = args.log or (args.meta_dir / "run.log" if args.meta_dir else None)
    if not cpu_path and not log_path:
        ap.error("укажите meta_dir или --cpu/--log")

    cpu_log = load_cpu_csv(cpu_path) if cpu_path and Path(cpu_path).exists() else []
    durations = (parse_run_log_durations(
        Path(log_path).read_text(encoding="utf-8", errors="replace"))
        if log_path and Path(log_path).exists() else [])

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

    stats = build_charts(cpu_log, durations, out_path)
    print(f"cpu.csv: {stats.get('cpu_samples', 0)} замеров"
          f" (avg {stats.get('cpu_avg', '—')}% / max {stats.get('cpu_max', '—')}%)")
    print(f"run.log: {stats.get('events', 0)} событий"
          f" (цикл движения avg {stats.get('dur_avg_work', '—')} с,"
          f" min {stats.get('dur_min_work', '—')} / max {stats.get('dur_max_work', '—')})")
    print(f"→ {stats['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
