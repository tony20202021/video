"""
Pipeline status — таблица сервисов + статистика за 10 мин.
Выводит на экран и сохраняет .md рядом.

Usage:
    python scripts/utils/pipeline_status.py
    ./sh/status_linux.sh
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

WINDOW_MIN = 10  # статистика за последние N минут

# ─── .env ─────────────────────────────────────────────────────────────────────

env_file = REPO / ".env"
if env_file.is_file():
    for _line in env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line and "<" not in _line:
            k, _, v = _line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

GROUPS_VER    = os.environ.get("GROUPS_VER", "v3")
RESIDENTS_VER = os.environ.get("RESIDENTS_VER", "v1")

# ─── описание сервисов ────────────────────────────────────────────────────────

SERVICES = [
    {
        "name":       "video-transfer",
        "input":      "Windows → HTTP POST :8765",
        "output":     ".output/transfer/diff/  service/",
        "dir":        REPO / ".output/transfer",
        "log_work":   r"POST|принят|received|Готово",
        "log_wait":   r"ожидание|нет файлов",
        "stats_pat":  r'"POST /file HTTP/1\.1" 200',  # принятые файлы
        "has_timing": False,
    },
    {
        "name":       "video-yolo",
        "input":      ".output/transfer/diff/  service/",
        "output":     ".output/pipeline/  2_yolo_boxes_files/images/",
        "dir":        REPO / ".output/pipeline/2_yolo_boxes_files/images",
        "log_work":   r"Готово",
        "log_wait":   r"ожидание|Файлов нет",
        "stats_pat":  r"Готово\. Время:",
        "has_timing": True,
    },
    {
        "name":       "video-classify",
        "input":      ".output/pipeline/  2_yolo_boxes_files/images/",
        "output":     f".data/groups/{GROUPS_VER}/inference/  images/{{date}}/{{class}}/",
        "dir":        REPO / f".data/groups/{GROUPS_VER}/inference/images",
        "log_work":   r"Готово|Найдено кропов",
        "log_wait":   r"ожидание|Кропов нет",
        "stats_pat":  r"Готово\. Время:",
        "has_timing": True,
    },
    {
        "name":       "video-identify",
        "input":      f".data/groups/{GROUPS_VER}/inference/  images/{{date}}/1_resident/ 4_guest/",
        "output":     f".data/residents/{RESIDENTS_VER}/inference/  images/{{date}}/{{person}}/",
        "dir":        REPO / f".data/residents/{RESIDENTS_VER}/inference/images",
        "log_work":   r"Готово",
        "log_wait":   r"ожидание|Кропов нет",
        "stats_pat":  r"Готово\. Время:",
        "has_timing": True,
    },
]

# ─── helpers ──────────────────────────────────────────────────────────────────

def svc_state(name: str) -> str:
    try:
        r = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def latest_file(directory: Path) -> tuple[str, str]:
    best_ts, best_path = 0.0, None
    try:
        for f in directory.rglob("*.jpg"):
            ts = f.stat().st_mtime
            if ts > best_ts:
                best_ts, best_path = ts, f
    except Exception:
        pass
    if best_path is None:
        return "—", "—"
    t = datetime.fromtimestamp(best_ts).strftime("%m-%d %H:%M")
    short = f"…/{best_path.parent.name}/{best_path.name}"
    return t, short


def _shorten_log(raw: str) -> str:
    if ": " in raw:
        msg = raw.split(": ", 1)[1]
    else:
        msg = raw
    msg = re.sub(r"\s+(DEBUG|INFO|WARNING|ERROR|ACCESS)\s+", "  ", msg)
    msg = re.sub(r"\d+\.\d+\.\d+\.\d+:\d+\s+-\s+", "", msg)
    msg = re.sub(r"Готово\. Время: ([\d.]+) с\..*", r"Готово \1с", msg)
    msg = re.sub(r'"(POST \S+) HTTP/[^"]*" (\d+) \w+', r"\1 \2", msg)
    msg = re.sub(r"\([\w_]+\) ", "", msg)
    msg = re.sub(r"— ожидание \d+s.*", "→ ожидание", msg)
    msg = re.sub(r" в /\S+.*", "", msg)
    return msg.strip()


def last_log(svc: str, pattern: str) -> str:
    try:
        r = subprocess.run(
            ["journalctl", "-u", svc, "--no-pager", "-n2000"],
            capture_output=True, text=True,
        )
        lines = [l for l in r.stdout.splitlines() if re.search(pattern, l)]
        return _shorten_log(lines[-1]) if lines else "—"
    except Exception:
        return "—"


def service_stats(svc: str, stats_pat: str, has_timing: bool) -> dict:
    """Статистика работы сервиса за последние WINDOW_MIN минут."""
    try:
        r = subprocess.run(
            ["journalctl", "-u", svc, "--no-pager",
             "--since", f"{WINDOW_MIN} minutes ago"],
            capture_output=True, text=True,
        )
        count = 0
        times: list[float] = []
        for line in r.stdout.splitlines():
            if re.search(stats_pat, line):
                count += 1
                if has_timing:
                    m = re.search(r"Время:\s*([\d.,]+)\s*с", line)
                    if m:
                        times.append(float(m.group(1).replace(",", ".")))
        mn   = round(min(times), 1) if times else None
        avg  = round(sum(times) / len(times), 1) if times else None
        mx   = round(max(times), 1) if times else None
        busy = round(min(sum(times) / (WINDOW_MIN * 60) * 100, 100.0), 1) if times else None
        return dict(count=count, min=mn, avg=avg, max=mx, busy=busy)
    except Exception:
        return dict(count=0, min=None, avg=None, max=None, busy=None)


def fmt_stats(st: dict, has_timing: bool) -> str:
    if st["count"] == 0:
        return "—"
    s = f"{st['count']}×"
    if has_timing and st["avg"] is not None:
        s += f"  {st['min']}/{st['avg']}/{st['max']}с"
    if st["busy"] is not None:
        s += f"  {st['busy']}%"
    return s


# ─── сбор данных ──────────────────────────────────────────────────────────────

def collect() -> list[dict]:
    rows = []
    for s in SERVICES:
        state     = svc_state(s["name"])
        file_time, file_name = latest_file(s["dir"])
        st        = service_stats(s["name"], s["stats_pat"], s["has_timing"])
        rows.append({
            "name":      s["name"],
            "input":     s["input"],
            "output":    s["output"],
            "state":     state,
            "file_time": file_time,
            "file_name": file_name,
            "log_work":  last_log(s["name"], s["log_work"]),
            "log_wait":  last_log(s["name"], s["log_wait"]),
            "stats":     fmt_stats(st, s["has_timing"]),
        })
    return rows


# ─── рендер: терминал ─────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    if "  " in text:
        part0, part1 = text.split("  ", 1)
        if len(part0) <= width:
            return [part0, part1[:width]]
    return [text[:width], text[width:2 * width]]


def _box(headers: list[str], rows_data: list[list[str]], widths: list[int]) -> list[str]:
    def hline(l: str, m: str, r: str) -> str:
        return l + m.join("─" * (w + 2) for w in widths) + r

    out = [hline("┌", "┬", "┐")]
    out.append("│ " + " │ ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " │")
    out.append(hline("├", "┼", "┤"))
    for row in rows_data:
        wrapped = [_wrap(str(v), widths[i]) for i, v in enumerate(row)]
        height  = max(len(c) for c in wrapped)
        if height == 1:
            cells = [str(v).ljust(widths[i]) for i, v in enumerate(row)]
            out.append("│ " + " │ ".join(cells) + " │")
        else:
            for line_i in range(height):
                cells = []
                for j, lines in enumerate(wrapped):
                    v = lines[line_i] if line_i < len(lines) else ""
                    cells.append(v.ljust(widths[j]))
                out.append("│ " + " │ ".join(cells) + " │")
        out.append(hline("├", "┼", "┤"))
    out[-1] = hline("└", "┴", "┘")
    return out


def render_term(rows: list[dict], ts: str) -> str:
    out = ["", f"  PIPELINE STATUS    {ts}", ""]

    headers = ["Сервер", "Сервис", "Статус", "Вход", "Выход", "Файл", "Лог: работа", "Лог: ожид.", "10м (N мин/ср/макс %)"]
    widths  = [7, 18, 8, 36, 42, 11, 28, 24, 26]
    data = [
        ["Linux",
         r["name"],
         ("✓" if r["state"] == "active" else "✗") + " " + r["state"],
         r["input"],
         r["output"],
         r["file_time"],
         r["log_work"],
         r["log_wait"],
         r["stats"]]
        for r in rows
    ]
    for line in _box(headers, data, widths):
        out.append("  " + line)

    out.append("")
    return "\n".join(out)


# ─── рендер: markdown ─────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows_data: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows_data:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return out


def render_md(rows: list[dict], ts: str) -> str:
    out = ["# Pipeline Status", "", f"_{ts}_", ""]

    headers = ["Сервер", "Сервис", "Статус", "Вход", "Выход", "Файл", "Лог: работа", "Лог: ожид.", "10м (N мин/ср/макс %)"]
    data = [
        ["Linux",
         f"`{r['name']}`",
         ("✓ " if r["state"] == "active" else "✗ ") + r["state"],
         r["input"].replace("  ", "<br>"),
         r["output"].replace("  ", "<br>"),
         r["file_time"],
         r["log_work"],
         r["log_wait"],
         r["stats"]]
        for r in rows
    ]
    out.extend(_md_table(headers, data))
    out.append("")

    return "\n".join(out)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = collect()

    term_out = render_term(rows, ts)
    md_out   = render_md(rows, ts)

    out_dir = REPO / ".output/status"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem    = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(md_out, encoding="utf-8")

    print(term_out)
    print(f"  → {md_path}")
    print()


if __name__ == "__main__":
    main()
