"""
Веб-интерфейс разметки кропов людей.

Открывает браузер с простым UI: показывает кроп, предлагает выбрать класс.
Результаты сохраняются в labels.json рядом с каталогом кропов.

Использование:
  python scripts/train/label_ui.py
  python scripts/train/label_ui.py --input .output/cameras/5_diff_yolo_boxes_low/run_XXX/crops
  python scripts/train/label_ui.py --port 5050

Горячие клавиши в браузере:
  1-5  — присвоить класс (1=resident, 2=courier, 3=delivery, 4=utilities, 5=other)
  →    — следующий без разметки
  ←    — предыдущий
  U    — пропустить (unknown)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INPUT   = REPO_ROOT / ".output" / "cameras" / "5_diff_yolo_boxes_low"
DEFAULT_LABELS  = REPO_ROOT / ".output" / "train" / "4_label_ui"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASSES = ["resident", "courier", "delivery", "utilities", "other"]

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Разметка кропов</title>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
  h2 { color: #7ec8e3; margin: 0 0 10px; }
  .container { display: flex; gap: 20px; align-items: flex-start; }
  .img-box { flex: 0 0 auto; }
  .img-box img { max-width: 400px; max-height: 400px; border: 2px solid #444; display: block; }
  .info { flex: 1; }
  .fname { color: #aaa; font-size: 12px; margin-bottom: 10px; word-break: break-all; }
  .progress { color: #7ec8e3; margin-bottom: 15px; font-size: 14px; }
  .current-label { font-size: 18px; margin-bottom: 15px; }
  .current-label span { color: #f9ca24; font-weight: bold; }
  .buttons { display: flex; flex-direction: column; gap: 8px; }
  button { padding: 10px 20px; font-size: 14px; cursor: pointer; border: none;
           border-radius: 4px; text-align: left; font-family: monospace; }
  .btn-class { background: #16213e; color: #eee; border-left: 4px solid #555; }
  .btn-class:hover, .btn-class.active { border-left-color: #f9ca24; background: #0f3460; color: #f9ca24; }
  .btn-nav  { background: #0f3460; color: #7ec8e3; }
  .btn-nav:hover { background: #1a4a7a; }
  .btn-skip { background: #2d1b1b; color: #e74c3c; }
  .btn-skip:hover { background: #4a2020; }
  .shortcuts { color: #666; font-size: 11px; margin-top: 15px; line-height: 1.6; }
  .labeled-count { color: #27ae60; }
  .nav-row { display: flex; gap: 8px; margin-top: 10px; }
</style>
</head>
<body>
<h2>Разметка кропов людей</h2>
<div id="app"></div>
<script>
let idx = 0;
let crops = [];
let labels = {};

async function loadState() {
  const r = await fetch('/api/state');
  const d = await r.json();
  crops = d.crops;
  labels = d.labels;
  idx = d.current_idx;
  render();
}

async function setLabel(cls) {
  if (idx >= crops.length) return;
  const r = await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({file: crops[idx], label: cls})
  });
  const d = await r.json();
  labels = d.labels;
  if (cls !== null) idx = Math.min(idx + 1, crops.length - 1);
  render();
}

async function navigate(delta) {
  idx = Math.max(0, Math.min(crops.length - 1, idx + delta));
  render();
}

function render() {
  if (!crops.length) {
    document.getElementById('app').innerHTML = '<p>Нет кропов для разметки.</p>';
    return;
  }
  const f = crops[idx];
  const lbl = labels[f] || '';
  const labeled = Object.keys(labels).length;

  const classButtons = [
    ['1', 'resident', '#27ae60'],
    ['2', 'courier', '#3498db'],
    ['3', 'delivery', '#e67e22'],
    ['4', 'utilities', '#9b59b6'],
    ['5', 'other', '#95a5a6'],
  ].map(([key, cls, color]) =>
    `<button class="btn-class${lbl===cls?' active':''}" onclick="setLabel('${cls}')"
      style="${lbl===cls?'border-left-color:'+color+';color:'+color:''}">[${key}] ${cls}</button>`
  ).join('');

  document.getElementById('app').innerHTML = `
    <div class="container">
      <div class="img-box">
        <img src="/image/${encodeURIComponent(f)}" alt="${f}" />
      </div>
      <div class="info">
        <div class="progress">
          Кадр <b>${idx+1}</b> / ${crops.length} &nbsp;|&nbsp;
          <span class="labeled-count">Размечено: ${labeled}</span>
        </div>
        <div class="fname">${f}</div>
        <div class="current-label">Класс: <span>${lbl || '—'}</span></div>
        <div class="buttons">${classButtons}</div>
        <div class="nav-row">
          <button class="btn-nav" onclick="navigate(-1)">← Назад</button>
          <button class="btn-nav" onclick="navigate(1)">Вперёд →</button>
          <button class="btn-skip" onclick="setLabel(null)">Пропустить (U)</button>
        </div>
        <div class="shortcuts">
          Горячие клавиши: 1-5 = классы | ← → = навигация | U = пропустить
        </div>
      </div>
    </div>`;
}

document.addEventListener('keydown', e => {
  const map = {'1':'resident','2':'courier','3':'delivery','4':'utilities','5':'other'};
  if (map[e.key]) { setLabel(map[e.key]); return; }
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
  if (e.key === 'u' || e.key === 'U') setLabel(null);
});

loadState();
</script>
</body>
</html>"""


def run_server(input_dir: Path, port: int, labels_path: Path) -> None:
    from flask import Flask, jsonify, request, send_file, Response

    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Собираем все кропы
    crops: list[str] = []
    search_dirs = [input_dir / "crops"] if (input_dir / "crops").is_dir() else []
    if not search_dirs:
        for run_dir in sorted(input_dir.iterdir()):
            sub = run_dir / "crops"
            if sub.is_dir():
                search_dirs.append(sub)

    for d in search_dirs:
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                crops.append(str(f.relative_to(REPO_ROOT)))

    # Загружаем существующую разметку
    labels: dict[str, str] = {}
    if labels_path.is_file():
        existing = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = existing.get("labels", {})

    # Начинаем с первого неразмеченного
    current = next((i for i, c in enumerate(crops) if c not in labels), 0)

    @app.route("/")
    def index():
        return Response(_HTML, mimetype="text/html")

    @app.route("/api/state")
    def state():
        return jsonify({"crops": crops, "labels": labels, "current_idx": current})

    @app.route("/api/label", methods=["POST"])
    def label():
        nonlocal current
        data = request.get_json()
        fname, cls = data["file"], data["label"]
        if cls is not None:
            labels[fname] = cls
        _save_labels(labels_path, labels)
        return jsonify({"labels": labels})

    @app.route("/image/<path:fname>")
    def image(fname: str):
        full = REPO_ROOT / fname
        if not full.is_file():
            return Response("not found", status=404)
        mime = mimetypes.guess_type(str(full))[0] or "image/jpeg"
        return send_file(str(full), mimetype=mime)

    url = f"http://127.0.0.1:{port}"
    print(f"Разметчик запущен: {url}")
    print(f"Кропов: {len(crops)}  Уже размечено: {len(labels)}")
    print(f"Метки: {labels_path}")
    print("Ctrl+C для остановки\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


def _save_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "labels": labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Веб-интерфейс разметки кропов")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Файл для сохранения меток (default: рядом с input/labels.json)")
    parser.add_argument("--port",   type=int, default=5050)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Нет каталога: {args.input}", file=sys.stderr)
        return 1

    labels_path = args.labels or (DEFAULT_LABELS / "labels.json")
    run_server(args.input, args.port, labels_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
