"""
Pipeline status — таблица сервисов + статистика за 10 мин.
Выводит на экран и сохраняет .md рядом.

Usage:
    python scripts/utils/pipeline_status.py
    ./sh/status/status_linux.sh
"""
from __future__ import annotations

import csv as _csv
import os
import re
import subprocess
from datetime import datetime, timedelta
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
        "in_label":   "Windows → HTTP :8765",
        "in_dir":     None,                      # приём по HTTP — локального входного каталога нет
        "out_label":  ".output/transfer/  diff/  service/",
        "out_dir":    REPO / ".output/transfer/diff",
        "log_work":   r"POST|Готово\. Время:",
        "log_wait":   r"\[recv\].*heartbeat",  # heartbeat-файл = камера idle, движения нет
        "stats_pat":  r"\[recv\].*Готово\. Время:",
        "has_timing": True,
    },
    {
        "name":       "video-yolo",
        "in_label":   ".output/transfer/  diff/  service/",
        "in_dir":     REPO / ".output/transfer/diff",
        "out_label":  ".output/pipeline/  2_yolo_boxes_files/images/",
        "out_dir":    REPO / ".output/pipeline/2_yolo_boxes_files/images",
        "log_work":   r"Готово",
        "log_wait":   r"ожидание|Файлов нет",
        "stats_pat":  r"Готово\. Время:",
        "has_timing": True,
    },
    {
        "name":       "video-classify",
        "in_label":   ".output/pipeline/  2_yolo_boxes_files/images/",
        "in_dir":     REPO / ".output/pipeline/2_yolo_boxes_files/images",
        "out_label":  f".data/groups/{GROUPS_VER}/inference/  images/",
        "out_dir":    REPO / f".data/groups/{GROUPS_VER}/inference/images",
        "log_work":   r"Готово|Найдено кропов",
        "log_wait":   r"ожидание|Кропов нет",
        "stats_pat":  r"Готово\. Время:",
        "has_timing": True,
    },
    {
        "name":       "video-identify",
        "in_label":   f"groups/{GROUPS_VER}/inference/  1_resident/  4_guest/",
        "in_dir":     REPO / f".data/groups/{GROUPS_VER}/inference/images",
        "out_label":  f".data/residents/{RESIDENTS_VER}/inference/  images/",
        "out_dir":    REPO / f".data/residents/{RESIDENTS_VER}/inference/images",
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


def dir_state(directory: Path | None) -> str:
    """Кол-во *.jpg + кол-во подкаталогов + дата-время последнего файла.

    '—' если файлов нет/каталог отсутствует/None — очередь не копится (пайплайн успевает).
    """
    if directory is None:
        return "—"
    files = 0
    dirs = 0
    best_ts = 0.0
    try:
        for root, ds, fs in os.walk(directory):
            dirs += len(ds)
            for f in fs:
                if f.endswith(".jpg"):
                    files += 1
                    try:
                        ts = os.path.getmtime(os.path.join(root, f))
                        if ts > best_ts:
                            best_ts = ts
                    except OSError:
                        pass
    except Exception:
        pass
    if files == 0:
        return "—"
    t = datetime.fromtimestamp(best_ts).strftime("%Y-%m-%d %H:%M")
    return f"{files} файл. · {dirs} кат.\n{t}"


def _env_val(key: str, default: str = "") -> str:
    """Значение из os.environ без инлайн-комментария и хвостовых пробелов."""
    v = os.environ.get(key, "")
    v = v.split("#")[0].strip()
    return v.split()[0] if v else default


def _ip_to_label() -> dict:
    """IP машины → метка (из .env TS_*)."""
    m = {}
    for ip_key, lbl_key in (("TS_DEVELOP", "TS_DEVELOP_LABEL"),
                            ("TS_CAMERAS_3", "TS_CAMERAS_3_LABEL"),
                            ("TS_CAMERAS_1", "TS_CAMERAS_1_LABEL"),
                            ("TS_SERVER", "")):
        ip = _env_val(ip_key)
        if ip:
            m[ip] = (os.environ.get(lbl_key, "").split("#")[0].strip() if lbl_key else "server") or ip
    return m


def transfer_in_label() -> str:
    """Вход video-transfer: реальные машины-клиенты из логов (POST /file),
    иначе адрес прослушивания TRANSFER_HOST:TRANSFER_PORT из .env."""
    port = _env_val("TRANSFER_PORT", "8765")
    clients = []
    try:
        r = subprocess.run(["journalctl", "-u", "video-transfer", "--no-pager", "-n", "3000"],
                           capture_output=True, text=True)
        ips = re.findall(r"(\d+\.\d+\.\d+\.\d+):\d+ - .*POST /file", r.stdout)
        m = _ip_to_label()
        clients = [m.get(ip, ip) for ip in dict.fromkeys(ips)]  # уникальные, сохраняя порядок
    except Exception:
        pass
    if clients:
        return "← " + ", ".join(clients) + f" · HTTP :{port}"
    host = _env_val("TRANSFER_HOST", "0.0.0.0")
    return f"← слушает {host}:{port}"


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


def _with_date(raw: str) -> str:
    """Строка лога с датой первой строкой (journalctl -o short-iso: '2026-07-18T…').
    Дата берётся из журнала — видно, если событие старое."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw)
    short = _shorten_log(raw)
    return f"{m.group(1)}\n{short}" if m else short


def last_log(svc: str, pattern: str, exclude_pat: str | None = None) -> str:
    try:
        r = subprocess.run(
            ["journalctl", "-u", svc, "--no-pager", "-o", "short-iso", "-n5000"],
            capture_output=True, text=True,
        )
        all_lines = r.stdout.splitlines()
        matched = [l for l in all_lines if re.search(pattern, l)]
        if matched:
            return _with_date(matched[-1])
        if exclude_pat:
            # fallback: строки не работа, содержат признак ожидания
            idle_hint = re.compile(r"нет|ожид|пуст|\bwait\b|\bidle\b|\bempty\b", re.IGNORECASE)
            candidates = [l for l in all_lines
                          if not re.search(exclude_pat, l)
                          and re.search(r"\d{2}:\d{2}:\d{2}", l)
                          and idle_hint.search(l)]
            if candidates:
                return _with_date(candidates[-1])
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
    return dict(count=count, min=mn, avg=avg, max=mx, last=last,
                window=window_min)


def service_stats(svc: str, stats_pat: str, has_timing: bool) -> dict:
    """Статистика: 10м → 60м → 24ч (первое ненулевое окно)."""
    _empty = dict(count=0, min=None, avg=None, max=None, last=None,
                  window=WINDOW_MIN)
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


def _cpu_csv_paths(svc_name: str, window_min: int) -> list[Path]:
    """Файлы cpu.csv сервиса за последние window_min минут."""
    today   = datetime.now().strftime("%Y%m%d")
    cutoff  = datetime.now().timestamp() - window_min * 60

    if svc_name == "video-transfer":
        p = REPO / ".output/transfer/meta" / today / "cpu.csv"
        return [p] if p.is_file() else []

    if svc_name == "video-yolo":
        p = REPO / ".output/pipeline/2_yolo_boxes_files/meta" / today / "cpu.csv"
        return [p] if p.is_file() else []

    if svc_name == "video-classify":
        base = REPO / f".data/groups/{GROUPS_VER}/inference/meta" / today
        if not base.is_dir():
            return []
        return [p for p in base.glob("*/cpu.csv") if p.stat().st_mtime >= cutoff]

    if svc_name == "video-identify":
        base = REPO / f".data/residents/{RESIDENTS_VER}/inference/meta" / today
        if not base.is_dir():
            return []
        return [p for p in base.glob("*/cpu.csv") if p.stat().st_mtime >= cutoff]

    return []


def _service_cpu_window(svc_name: str, window_min: int) -> str | None:
    """ЦПУ из cpu.csv за последние window_min минут → мин%/ср%/макс%/посл%."""
    paths = _cpu_csv_paths(svc_name, window_min)
    if not paths:
        return None

    cutoff  = datetime.now() - timedelta(minutes=window_min)
    samples: list[int] = []

    for path in sorted(paths):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    try:
                        ts = datetime.strptime(row["ts_msk"][:15], "%Y%m%d_%H%M%S")
                        if ts >= cutoff:
                            samples.append(round(float(row["cpu_pct"])))
                    except (ValueError, KeyError):
                        pass
        except Exception:
            pass

    if not samples:
        return None

    mn   = min(samples)
    avg  = round(sum(samples) / len(samples))
    mx   = max(samples)
    last = samples[-1]
    return f"{mn}%/{avg}%/{mx}%/{last}%"


def service_cpu(svc_name: str, window_min: int = WINDOW_MIN) -> str | None:
    """ЦПУ с тем же фолбэком окон, что и статистика кадров (10м → 60м → 24ч),
    чтобы CPU показывался у всех сервисов, а не только у непрерывно пишущих cpu.csv."""
    for w in (window_min, 60, 1440):
        r = _service_cpu_window(svc_name, w)
        if r:
            return r
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
    if cpu_line is not None:
        lines.append(cpu_line)
    return "\n".join(lines)


# ─── сбор данных ──────────────────────────────────────────────────────────────

def collect() -> list[dict]:
    rows = []
    for s in SERVICES:
        state     = svc_state(s["name"])
        in_state  = dir_state(s["in_dir"])
        out_state = dir_state(s["out_dir"])
        in_label  = transfer_in_label() if s["name"] == "video-transfer" else s["in_label"]
        st        = service_stats(s["name"], s["stats_pat"], s["has_timing"])
        cpu_line  = service_cpu(s["name"], st.get("window", WINDOW_MIN)) if state == "active" else None
        rows.append({
            "name":      s["name"],
            "input":     _fmt_tree(in_label) + "\n" + in_state,
            "output":    _fmt_tree(s["out_label"]) + "\n" + out_state,
            "state":     state,
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


_NOWRAP = "\033[?7l"   # отключить перенос строк в терминале
_WRAP   = "\033[?7h"   # включить обратно


def render_term(rows: list[dict], ts: str) -> str:
    out = [_NOWRAP, "", f"  PIPELINE STATUS    {ts}", ""]

    headers = ["Сервер", "Сервис", "Статус", "Вход\n(N файл. + послед.)", "Выход\n(N файл. + послед.)",
               "Лог: работа", "Лог: ожид.",
               "N кадров (за 10м/1ч/24ч)\n1кадр (мин/ср/макс/посл)\nцпу% (мин/ср/макс/посл)"]
    widths  = [7, 20, 8, 38, 38, 150, 120, 65]
    data = [
        ["Linux",
         r["name"],
         ("✓" if r["state"] == "active" else "✗") + " " + r["state"],
         r["input"],
         r["output"],
         r["log_work"],
         r["log_wait"],
         r["stats"]]
        for r in rows
    ]
    for line in _box(headers, data, widths):
        out.append("  " + line)

    out.append(_WRAP)
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

    headers = ["Сервер", "Сервис", "Статус", "Вход<br>(N файл. + послед.)", "Выход<br>(N файл. + послед.)",
               "Лог: работа", "Лог: ожид.",
               "N кадров (за 10м/1ч/24ч)<br>1кадр (мин/ср/макс/посл)<br>цпу% (мин/ср/макс/посл)"]
    data = [
        ["Linux",
         f"`{r['name']}`",
         ("✓ " if r["state"] == "active" else "✗ ") + r["state"],
         r["input"].replace("\n", "<br>"),
         r["output"].replace("\n", "<br>"),
         r["log_work"].replace("\n", "<br>"),
         r["log_wait"].replace("\n", "<br>"),
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
