"""Веб-интерфейс — HTML страница с мониторингом системы."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from backend.routers.status import full_status

router = APIRouter(tags=["web"])

_CSS = """
body{font-family:monospace;padding:1.5em;background:#1a1a1a;color:#e0e0e0;max-width:900px;margin:0 auto}
h1{color:#4fc3f7;margin-bottom:0}
h2{color:#81d4fa;margin-top:1.5em;border-bottom:1px solid #333;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin-top:.5em}
th,td{padding:5px 10px;text-align:left;border-bottom:1px solid #2a2a2a}
th{color:#4fc3f7}
.ok{color:#a5d6a7}.warn{color:#ffcc80}.err{color:#ef9a9a}
.meta{color:#888;font-size:.85em}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.8em;background:#263238}
a{color:#4fc3f7;text-decoration:none}
"""

_JS = """
setTimeout(()=>location.reload(), 30000);
"""


def _disk_rows(disks: list) -> str:
    rows = []
    for d in disks:
        pct = d["used_pct"]
        cls = "err" if pct > 90 else ("warn" if pct > 75 else "ok")
        rows.append(
            f"<tr><td>{d['mount']}</td><td>{d['total_gb']} GB</td>"
            f"<td>{d['used_gb']} GB</td><td>{d['free_gb']} GB</td>"
            f"<td class='{cls}'>{pct}%</td></tr>"
        )
    return "".join(rows)


def _proc_rows(procs: list, key: str) -> str:
    rows = []
    for p in procs:
        val = f"{p[key]} MB" if "ram" in key else f"{p[key]}%"
        rows.append(f"<tr><td>{p['name']}</td><td>{val}</td></tr>")
    return "".join(rows)


def _status_section(data: dict) -> str:
    cpu = data.get("cpu_pct", "—")
    ram_pct = data.get("ram_pct", "—")
    ram_used = data.get("ram_used_mb", "—")
    ram_total = data.get("ram_total_mb", "—")
    disk_rows = _disk_rows(data.get("disks", []))
    top_cpu_rows = _proc_rows(data.get("top_cpu", []), "cpu_pct")
    top_ram_rows = _proc_rows(data.get("top_ram", []), "ram_mb")

    return f"""
<h2>Система</h2>
<p>CPU: <b>{cpu}%</b> &nbsp;|&nbsp; RAM: <b>{ram_used}/{ram_total} MB ({ram_pct}%)</b>
   &nbsp;|&nbsp; uptime: {data.get('uptime_sec', 0)} s
   &nbsp;|&nbsp; {data.get('platform', '')} / Python {data.get('python', '')}</p>

<table>
  <tr><th>Раздел</th><th>Всего</th><th>Занято</th><th>Свободно</th><th>%</th></tr>
  {disk_rows}
</table>

<div style="display:flex;gap:2em;margin-top:1em">
  <div>
    <b>Top CPU</b>
    <table><tr><th>Процесс</th><th>CPU%</th></tr>{top_cpu_rows}</table>
  </div>
  <div>
    <b>Top RAM</b>
    <table><tr><th>Процесс</th><th>MB</th></tr>{top_ram_rows}</table>
  </div>
</div>
"""


async def _cameras_section() -> str:
    try:
        from backend.routers.cameras import _collect_cameras
        cameras = _collect_cameras()
    except Exception:
        return "<h2>Камеры</h2><p class='err'>Ошибка загрузки камер</p>"

    if not cameras:
        return "<h2>Камеры</h2><p class='meta'>Нет настроенных камер</p>"

    rows = []
    for cam in cameras:
        icon = "🟢" if cam.get("online") else "🔴"
        rows.append(
            f"<tr><td>{icon}</td><td>{cam.get('camera_id', '—')}</td>"
            f"<td>{cam.get('host', '—')}</td></tr>"
        )
    return f"""
<h2>Камеры</h2>
<table>
  <tr><th></th><th>ID</th><th>Хост</th></tr>
  {"".join(rows)}
</table>
"""


async def _events_section() -> str:
    try:
        from backend.routers.events import list_events
        result = await list_events(skip=0, limit=10)
        items = result.get("data", [])
        total = result.get("total", 0)
    except Exception:
        return "<h2>События</h2><p class='meta'>MongoDB недоступна</p>"

    if not items:
        return "<h2>События</h2><p class='meta'>Событий нет</p>"

    rows = []
    for e in items:
        ts = str(e.get("timestamp", ""))[:19].replace("T", " ")
        cam = e.get("camera_id", "—")
        cls = e.get("group_class") or "—"
        person = e.get("person_id") or "—"
        eid = str(e.get("_id", ""))[:8]
        rows.append(
            f"<tr><td class='meta'>{ts}</td><td>{cam}</td>"
            f"<td>{cls}</td><td>{person}</td><td class='meta'>{eid}…</td></tr>"
        )

    return f"""
<h2>Последние события <span class='meta'>(всего: {total})</span></h2>
<table>
  <tr><th>Время</th><th>Камера</th><th>Класс</th><th>Житель</th><th>ID</th></tr>
  {"".join(rows)}
</table>
"""


@router.get("/", response_class=HTMLResponse)
async def index():
    data = full_status()
    status_section = _status_section(data)
    cameras_section = await _cameras_section()
    events_section = await _events_section()

    ts = data.get("ts_msk", "")
    status_cls = "ok" if data.get("status") == "ok" else "err"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Video Surveillance</title>
  <style>{_CSS}</style>
  <script>{_JS}</script>
</head>
<body>
  <h1>Video Surveillance</h1>
  <p class="{status_cls} meta">{ts} &nbsp;|&nbsp; обновление каждые 30 с</p>
  {status_section}
  {cameras_section}
  {events_section}
</body>
</html>"""
    return HTMLResponse(html)
