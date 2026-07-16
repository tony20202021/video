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
        "output":     ".output/transfer/  diff/  service/",
        "dir":        REPO / ".output/transfer",
        "log_work":   r"POST|Готово\. Время:",
        "log_wait":   r"\[recv\].*heartbeat",  # heartbeat-файл = камера idle, движения нет
        "stats_pat":  r"\[recv\].*Готово\. Время:",
        "has_timing": True,
    },
    {
        "name":       "video-yolo",
        "input":      ".output/transfer/  diff/  service/",
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
        "log_wait":   r"Идентификаций не найдено|ожидание|Кропов нет",
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
    t = datetime.fromtimestamp(best_ts).strftime("%Y-%m-%d\n%H:%M")
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
    # video-transfer [recv]: сокращаем длинный путь к heartbeat
    msg = re.sub(r"\[recv\] \S+/(cam_\d+_\d+_[a-z]+)\S*", r"[recv] \1", msg)
    # Убрать размер файла "(51.2 КБ)" и "meta=..." после [recv] cam_X
    msg = re.sub(r"\s+meta=\S+", "", msg)
    msg = re.sub(r"(\[recv\]\s+\S+)\s+\([\d.,]+\s*\S+\)", r"\1", msg)
    return msg.strip()


def last_log(svc: str, pattern: str, exclude_pat: str | None = None) -> str:
    try:
        r = subprocess.run(
            ["journalctl", "-u", svc, "--no-pager", "-n5000"],
            capture_output=True, text=True,
        )
        all_lines = r.stdout.splitlines()
        matched = [l for l in all_lines if re.search(pattern, l)]
        if matched:
            return _shorten_log(matched[-1])
        if exclude_pat:
            # fallback: строки не работа, содержат признак ожидания
            idle_hint = re.compile(r"нет|ожид|пуст|\bwait\b|\bidle\b|\bempty\b", re.IGNORECASE)
            candidates = [l for l in all_lines
                          if not re.search(exclude_pat, l)
                          and re.search(r"\d{2}:\d{2}:\d{2}", l)
                          and idle_hint.search(l)]
            if candidates:
                return _shorten_log(candidates[-1])
        return "—"
    except Exception:
        return "—"


def _stats_for_window(svc: str, stats_pat: str, has_timing: bool, window_min: int) -> dict:
    r = subprocess.run(
        ["journalctl", "-u", svc, "--no-pager",
         "--since", f"{window_min} minutes ago"],
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
    last = round(times[-1], 1) if times else None
    if times and count > 0:
        avg_iv = window_min * 60 / count
        per    = [t / avg_iv * 100 for t in times]
        bp_mn  = round(min(per))
        bp_avg = round(sum(per) / len(per))
        bp_mx  = round(max(per))
        bp_lst = round(per[-1])
    else:
        bp_mn = bp_avg = bp_mx = bp_lst = None
    return dict(count=count, min=mn, avg=avg, max=mx, last=last,
                bp_mn=bp_mn, bp_avg=bp_avg, bp_mx=bp_mx, bp_lst=bp_lst,
                window=window_min)


def service_stats(svc: str, stats_pat: str, has_timing: bool) -> dict:
    """Статистика: 10м → 60м → 24ч (первое ненулевое окно)."""
    _empty = dict(count=0, min=None, avg=None, max=None, last=None,
                  bp_mn=None, bp_avg=None, bp_mx=None, bp_lst=None, window=WINDOW_MIN)
    try:
        for window in [WINDOW_MIN, 60, 1440]:
            st = _stats_for_window(svc, stats_pat, has_timing, window)
            if st["count"] > 0:
                return st
        return _empty
    except Exception:
        return _empty


def _fmt_tree(text: str) -> str:
    """'A  B  C' → 'A\\n├─ B\\n└─ C'; 'A  B' → 'A\\nB' (один дочерний — без символа)."""
    parts = text.split("  ")
    if len(parts) == 1:
        return text
    parent, children = parts[0], parts[1:]
    if len(children) == 1:
        return parent + "\n" + children[0]
    lines = [parent]
    for i, child in enumerate(children):
        lines.append(("└─ " if i == len(children) - 1 else "├─ ") + child)
    return "\n".join(lines)


def service_cpu(svc_name: str) -> str | None:
    """Текущая загрузка ЦПУ процесса (один замер 0.3с)."""
    try:
        import time
        import psutil
        r = subprocess.run(
            ["systemctl", "show", "--property=MainPID", svc_name],
            capture_output=True, text=True,
        )
        pid = int(r.stdout.strip().split("=")[1])
        if pid <= 0:
            return None
        p = psutil.Process(pid)
        p.cpu_percent()  # warm-up
        time.sleep(0.3)
        pct = round(p.cpu_percent())
        return f"{pct}%"
    except Exception:
        return None


def fmt_stats(st: dict, has_timing: bool, cpu_line: str | None = None) -> str:
    if st["count"] == 0:
        return "—"
    w = st.get("window", WINDOW_MIN)
    if w >= 1440:
        suffix = " (24ч)"
    elif w >= 60:
        suffix = f" ({w // 60}ч)" if w % 60 == 0 else f" ({w}м)"
    else:
        suffix = f" ({w}м)"
    lines = [f"{st['count']}×{suffix}"]
    if has_timing and st["avg"] is not None:
        mn, avg, mx, lst = st["min"], st["avg"], st["max"], st["last"]
        lines.append(f"{mn}с/{avg}с/{mx}с/{lst}с")
        if st["bp_avg"] is not None:
            bp = st["bp_mn"], st["bp_avg"], st["bp_mx"], st["bp_lst"]
            lines.append(f"{bp[0]}%/{bp[1]}%/{bp[2]}%/{bp[3]}%")
    if cpu_line is not None:
        lines.append(cpu_line)
    return "\n".join(lines)


# ─── сбор данных ──────────────────────────────────────────────────────────────

def collect() -> list[dict]:
    rows = []
    for s in SERVICES:
        state     = svc_state(s["name"])
        file_time, file_name = latest_file(s["dir"])
        st        = service_stats(s["name"], s["stats_pat"], s["has_timing"])
        cpu_line  = service_cpu(s["name"]) if state == "active" else None
        rows.append({
            "name":      s["name"],
            "input":     s["input"],
            "output":    s["output"],
            "state":     state,
            "file_time": file_time,
            "file_name": file_name,
            "log_work":  last_log(s["name"], s["log_work"]),
            "log_wait":  last_log(s["name"], s["log_wait"], exclude_pat=s["log_work"]),
            "stats":     fmt_stats(st, s["has_timing"], cpu_line),
        })
    return rows


# ─── рендер: терминал ─────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    if "\n" in text:
        return text.split("\n")
    return [text]


def _box(headers: list[str], rows_data: list[list[str]], widths: list[int]) -> list[str]:
    def hline(l: str, m: str, r: str) -> str:
        return l + m.join("─" * (w + 2) for w in widths) + r

    header_lines = [h.split("\n") for h in headers]
    max_h = max(len(hl) for hl in header_lines)

    out = [hline("┌", "┬", "┐")]
    for line_i in range(max_h):
        cells = []
        for j, hl in enumerate(header_lines):
            v = hl[line_i] if line_i < len(hl) else ""
            cells.append(v.ljust(widths[j]))
        out.append("│ " + " │ ".join(cells) + " │")
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

    headers = ["Сервер", "Сервис", "Статус", "Вход", "Выход", "Файл", "Лог: работа", "Лог: ожид.",
               "за 10м:\nN× / с\n(мин/ср/макс/посл)\n%(мин/ср/макс/посл)\nцпу%"]
    widths  = [7, 18, 8, 40, 42, 20, 28, 24, 22]
    data = [
        ["Linux",
         r["name"],
         ("✓" if r["state"] == "active" else "✗") + " " + r["state"],
         _fmt_tree(r["input"]),
         _fmt_tree(r["output"]),
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

    headers = ["Сервер", "Сервис", "Статус", "Вход", "Выход", "Файл", "Лог: работа", "Лог: ожид.",
               "за 10м:<br>N×<br>с (мин/ср/макс/посл)<br>% (мин/ср/макс/посл)<br>цпу%"]
    data = [
        ["Linux",
         f"`{r['name']}`",
         ("✓ " if r["state"] == "active" else "✗ ") + r["state"],
         _fmt_tree(r["input"]).replace("\n", "<br>"),
         _fmt_tree(r["output"]).replace("\n", "<br>"),
         r["file_time"].replace("\n", "<br>"),
         r["log_work"],
         r["log_wait"],
         r["stats"].replace("\n", "<br>")]
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

    now     = datetime.now()
    out_dir = REPO / ".output/status" / now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem    = now.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(md_out, encoding="utf-8")

    print(term_out)
    print(f"  → {md_path}")
    print()


if __name__ == "__main__":
    main()
