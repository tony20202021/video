"""
Веб-интерфейс разметки кропов людей.

Открывает браузер с простым UI: показывает кроп, предлагает выбрать класс.
Результаты сохраняются в labels.json рядом с каталогом кропов.

Использование:
  python scripts/train/label_ui.py
  python scripts/train/label_ui.py --input .output/cameras/5_diff_yolo_boxes_low/run_XXX/crops
  python scripts/train/label_ui.py --port 8750

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
import os
import re
import sys
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.access import IpAllowlist, is_ip_allowed, log_ip_denied, parse_allowed_ips
from common.utils.classes import EXTRA_DATASET_DIRS, GROUP_CLASSES

_DEFAULT_PORT = int(os.environ.get("LABEL_UI_PORT", "8750"))

DEFAULT_INPUT   = REPO_ROOT / ".output" / "pipeline" / "2_yolo_boxes_files"
DEFAULT_LABELS  = REPO_ROOT / ".output" / "train" / "2_label_ui"
DEFAULT_DATASET = REPO_ROOT / ".data" / "groups"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_EXTRA_CLASSES = set(EXTRA_DATASET_DIRS)


def _load_classes(dataset_dir: Path | None) -> tuple[list[str], Path]:
    """Читает классы из папок датасета (sorted, без skip/unknown/new).
    Возвращает (classes, resolved_dataset_dir)."""
    if dataset_dir is None:
        versions = sorted(DEFAULT_DATASET.glob("v*/dataset.json"))
        if not versions:
            print("[!] Датасет не найден. Укажите --dataset .data/groups/v1", file=sys.stderr)
            sys.exit(1)
        dataset_dir = versions[-1].parent
    classes = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and d.name not in _EXTRA_CLASSES
    )
    if not classes:
        classes = list(GROUP_CLASSES)
    return classes, dataset_dir

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Разметка кропов</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: monospace; background: #1a1a2e; color: #eee; margin: 0;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  h2 { color: #7ec8e3; margin: 0; padding: 8px 16px; flex: 0 0 auto;
       border-bottom: 1px solid #2a2a4a; font-size: 15px;
       display: flex; align-items: center; gap: 16px; }
  h2 a { color: #7ec8e3; font-size: 12px; opacity: 0.6; text-decoration: none; }
  h2 a:hover { opacity: 1; }
  #app { flex: 1; overflow: hidden; }
  .container { display: flex; height: 100%; }

  /* ── левая панель (кнопки) ── */
  .panel-left { flex: none; width: 360px; min-width: 160px; padding: 14px 16px; overflow-y: auto;
                border-right: 1px solid #2a2a4a; display: flex; flex-direction: column; gap: 10px; }
  .drag-handle { flex: none; width: 5px; cursor: col-resize; background: transparent;
                 border-right: 1px solid #2a2a4a; transition: background .15s; }
  .drag-handle:hover { background: #7ec8e3; }
  .progress { color: #7ec8e3; font-size: 13px; }
  .labeled-count { color: #27ae60; }
  .fname { color: #777; font-size: 11px; word-break: break-all; }
  .current-label { font-size: 16px; }
  .current-label span { color: #f9ca24; font-weight: bold; }
  .buttons { display: flex; flex-direction: column; gap: 6px; }
  button { padding: 9px 14px; font-size: 13px; cursor: pointer; border: none;
           border-radius: 4px; text-align: left; font-family: monospace; width: 100%; }
  .btn-class { background: #16213e; color: #eee; border-left: 4px solid #555; }
  .btn-class:hover, .btn-class.active { border-left-color: #f9ca24; background: #0f3460; color: #f9ca24; }
  .nav-row { display: flex; gap: 6px; }
  .nav-row button { width: auto; flex: 1; }
  .btn-nav  { background: #0f3460; color: #7ec8e3; }
  .btn-nav:hover { background: #1a4a7a; }
  .btn-skip { background: #2d1b1b; color: #e74c3c; }
  .btn-skip:hover { background: #4a2020; }
  .shortcuts { color: #555; font-size: 10px; line-height: 1.7; }

  /* ── правая панель (картинка) ── */
  .panel-right { flex: 1; display: flex; align-items: center; justify-content: center;
                 overflow: hidden; background: #111120; }
  .panel-right img { max-width: 100%; max-height: 100%; object-fit: contain;
                     display: block; border: 2px solid #333; }
</style>
</head>
<body>
<h2>Разметка кропов людей
  <a href="/gallery">⊞ Галерея</a>
  <a href="/gallery/dataset">⊞ Датасет</a>
</h2>
<div id="app"></div>
<script>
const CLASS_COLORS = ["#27ae60","#e67e22","#9b59b6","#95a5a6","#3498db",
                      "#e74c3c","#1abc9c","#f39c12","#8e44ad"];
let idx = 0;
let crops = [];
let labels = {};
let classes = [];
let probs = {};
let panelW = 360;

function _attachDrag() {
  const handle = document.getElementById('drag-handle');
  const panel  = document.getElementById('panel-left');
  if (!handle || !panel) return;
  panel.style.width = panelW + 'px';
  handle.onmousedown = e => {
    const startX = e.clientX, startW = panel.offsetWidth;
    const onMove = e => {
      panelW = Math.max(160, Math.min(window.innerWidth * 0.7, startW + e.clientX - startX));
      panel.style.width = panelW + 'px';
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cssText = '';
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };
}

async function loadState() {
  const r = await fetch('/api/state');
  const d = await r.json();
  crops = d.crops;
  labels = d.labels;
  idx = d.current_idx;
  classes = d.classes;
  probs = d.probs || {};
  const p = new URLSearchParams(location.search).get('idx');
  if (p !== null) idx = Math.max(0, Math.min(crops.length - 1, parseInt(p)));
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
  idx = Math.min(idx + 1, crops.length - 1);
  render();
}

async function navigate(delta) {
  idx = Math.max(0, Math.min(crops.length - 1, idx + delta));
  render();
}

function render() {
  if (!crops.length) {
    document.getElementById('app').innerHTML = '<p style="padding:20px">Нет кропов для разметки.</p>';
    return;
  }
  const f = crops[idx];
  const lbl = labels[f] || '';
  const labeled = Object.keys(labels).length;

  const classButtons = classes.map((cls, i) => {
    const key = String(i + 1);
    const color = CLASS_COLORS[i] || '#888888';
    const active = lbl === cls;
    return `<button class="btn-class${active?' active':''}" onclick="setLabel('${cls}')"
      style="${active?'border-left-color:'+color+';color:'+color:''}">[${key}] ${cls}</button>`;
  }).join('');

  const shortcuts = classes.map((cls, i) => `${i+1}=${cls}`).join('<br>');

  document.getElementById('app').innerHTML = `
    <div class="container">
      <div class="panel-left" id="panel-left">
        <div class="progress">
          Кадр <b>${idx+1}</b> / ${crops.length}<br>
          <span class="labeled-count">Размечено: ${labeled}</span>
        </div>
        <div class="fname">${f}</div>
        <div class="current-label">Класс: <span>${lbl || '—'}</span></div>
        ${probBars(f)}
        <div class="buttons">${classButtons}</div>
        <div class="nav-row">
          <button class="btn-nav" onclick="navigate(-1)">← Назад</button>
          <button class="btn-nav" onclick="navigate(1)">Вперёд →</button>
        </div>
        <button class="btn-skip" onclick="setLabel('skip')">Пропустить (U)</button>
        <div class="shortcuts">
          ${shortcuts}<br>← → = навигация<br>U = пропустить
        </div>
      </div>
      <div class="drag-handle" id="drag-handle"></div>
      <div class="panel-right">
        <img src="/image/${encodeURIComponent(f)}" alt="${f}" />
      </div>
    </div>`;
  _attachDrag();
}

function probBars(filename) {
  const fname = filename.split('/').pop();
  const p = probs[fname];
  if (!p) return '';
  const rows = classes.map((cls, i) => {
    const color = CLASS_COLORS[i] || '#888888';
    const val = p[cls] !== undefined ? p[cls] : 0;
    const pct = (val * 100).toFixed(0);
    return `<div style="display:flex;align-items:center;gap:5px;margin:2px 0">
      <span style="flex:0 0 40%;min-width:0;font-size:10px;color:${color};word-break:break-all;line-height:1.3">${cls}</span>
      <div style="flex:1;min-width:30px;height:8px;background:#111120;border-radius:2px;align-self:center">
        <div style="width:${pct}%;height:100%;background:${color}bb;border-radius:2px"></div>
      </div>
      <span style="flex:0 0 38px;font-size:10px;color:#888;text-align:right">${val.toFixed(3)}</span>
    </div>`;
  }).join('');
  return `<div style="border-top:1px solid #2a2a4a;padding-top:6px;margin-top:4px">${rows}</div>`;
}

document.addEventListener('keydown', e => {
  const n = parseInt(e.key);
  if (!isNaN(n) && n >= 1 && n <= classes.length) { setLabel(classes[n-1]); return; }
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
  if (e.key === 'u' || e.key === 'U') setLabel('skip');
});

loadState();
</script>
</body>
</html>"""

_GALLERY_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Галерея кропов</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: monospace; background: #1a1a2e; color: #eee; margin: 0; }
  h2 { color: #7ec8e3; margin: 0; padding: 8px 16px; font-size: 15px;
       border-bottom: 1px solid #2a2a4a; display: flex; align-items: center; gap: 16px;
       position: sticky; top: 0; background: #1a1a2e; z-index: 10; }
  h2 a { color: #7ec8e3; font-size: 12px; opacity: 0.6; text-decoration: none; }
  h2 a:hover { opacity: 1; }
  h2 .stats { margin-left: auto; font-size: 12px; color: #555; }
  .section { padding: 12px 16px 4px; }
  .section-title { font-size: 13px; font-weight: bold; margin-bottom: 8px;
                   padding: 4px 10px; border-radius: 4px; display: inline-block; }
  :root { --tw: 120px; }
  .grid { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .tile { width: var(--tw); cursor: pointer; border: 2px solid #2a2a4a; border-radius: 4px;
          overflow: hidden; background: #111120; transition: border-color .15s; position: relative; }
  .tile:hover { border-color: #7ec8e3; }
  .tile.selected { border-color: #7ec8e3 !important; background: #0f2040; }
  .tile-check { position: absolute; top: 4px; left: 4px; z-index: 5; width: 20px; height: 20px;
                background: rgba(0,0,0,.65); border: 2px solid #556; border-radius: 3px;
                cursor: pointer; display: flex; align-items: center; justify-content: center;
                font-size: 13px; opacity: 0; transition: opacity .1s, border-color .1s; }
  .tile:hover .tile-check, .tile.selected .tile-check { opacity: 1; }
  .tile.selected .tile-check { border-color: #7ec8e3; color: #7ec8e3; background: rgba(0,30,60,.85); }
  .tile img { width: var(--tw); height: calc(var(--tw) * 0.75); object-fit: contain; display: block;
              background: #0a0a18; }
  .tile .tile-label { font-size: 9px; color: #777; padding: 3px 4px;
                      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85);
           z-index: 100; align-items: center; justify-content: center; flex-direction: column; gap: 12px; }
  #modal.open { display: flex; }
  #modal img { max-width: 90vw; max-height: 70vh; object-fit: contain; border: 2px solid #444; }
  #modal .modal-fname { color: #aaa; font-size: 11px; }
  #modal .modal-probs { width: 60vw; min-width: 300px; max-width: 560px; }
  #modal .modal-btns { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  #modal button { padding: 8px 16px; font-size: 13px; cursor: pointer; border: none;
                  border-radius: 4px; font-family: monospace; }
  #modal .btn-class { background: #16213e; color: #eee; border-left: 4px solid #555; }
  #modal .btn-class.active { border-left-color: #f9ca24; background: #0f3460; color: #f9ca24; }
  #modal .btn-go { background: #0f3460; color: #7ec8e3; }
  #modal .btn-close { background: #2a2a4a; color: #aaa; }
  .zoom-btn { background: #2a2a4a; color: #aaa; border: none; border-radius: 3px;
              cursor: pointer; font-size: 12px; padding: 2px 8px; font-family: monospace; }
  .zoom-btn:hover { background: #3a3a6a; color: #eee; }
  #bulk-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #0d1522;
              border-top: 2px solid #7ec8e3; padding: 8px 16px; display: none;
              align-items: center; gap: 8px; z-index: 50; flex-wrap: wrap; }
  #bulk-bar.show { display: flex; }
  #bulk-count { color: #7ec8e3; font-size: 13px; white-space: nowrap; min-width: 90px; }
  #bulk-btns { display: flex; gap: 6px; flex-wrap: wrap; }
  .btn-bulk { padding: 5px 11px; font-size: 12px; cursor: pointer; border: none;
              border-radius: 4px; font-family: monospace; }
  .btn-bulk-cls { background: #16213e; color: #eee; border-left: 3px solid #555; }
  .btn-bulk-cls:hover { background: #0f3460; color: #f9ca24; }
  .btn-skip { background: #2d1b1b; color: #e74c3c; }
  .btn-skip:hover { background: #4a2020; color: #ff6b6b; }
  .btn-bulk-skip { background: #2d1b1b; color: #e74c3c; }
  .btn-bulk-skip:hover { background: #4a2020; color: #ff6b6b; }
  .btn-bulk-desel { background: #2a2a4a; color: #999; margin-left: auto; }
  .btn-bulk-desel:hover { color: #eee; }
  #gallery { padding-bottom: 0; }
</style>
</head>
<body>
<h2>Галерея сессии
  <a href="/">← Разметка</a>
  <a href="/gallery/dataset">⊞ Датасет</a>
  <span style="display:flex;gap:4px;align-items:center">
    <button class="zoom-btn" onclick="selectAll()">☑ Все</button>
    <button class="zoom-btn" onclick="deselectAll()">☐ Снять</button>
    <button class="zoom-btn" onclick="zoom(-1)">−</button>
    <button class="zoom-btn" onclick="zoom(+1)">+</button>
  </span>
  <span class="stats" id="stats"></span>
</h2>
<div id="gallery"></div>

<div id="bulk-bar">
  <span id="bulk-count">Выбрано: 0</span>
  <div id="bulk-btns"></div>
  <button class="btn-bulk btn-bulk-desel" onclick="deselectAll()">✕ Снять</button>
</div>

<div id="modal">
  <img id="modal-img" src="" alt="">
  <div class="modal-fname" id="modal-fname"></div>
  <div class="modal-probs" id="modal-probs"></div>
  <div class="modal-btns" id="modal-btns"></div>
</div>

<script>
const CLASS_COLORS = ["#27ae60","#e67e22","#9b59b6","#95a5a6","#3498db",
                      "#e74c3c","#1abc9c","#f39c12","#8e44ad"];
let crops = [], labels = {}, classes = [];
let probs = {};
let modalFile = '', modalIdx = 0;
const ZOOM_STEPS = [60, 90, 120, 180, 240, 360];
let zoomIdx = 2;
let selected = new Set();
let lastToggleIdx = -1;
let flatOrder = [];  // crop indices in visual (DOM) order, rebuilt each render()

function zoom(d) {
  zoomIdx = Math.max(0, Math.min(ZOOM_STEPS.length - 1, zoomIdx + d));
  document.documentElement.style.setProperty('--tw', ZOOM_STEPS[zoomIdx] + 'px');
}

async function loadState() {
  const d = await (await fetch('/api/state')).json();
  crops = d.crops; labels = d.labels; classes = d.classes;
  probs = d.probs || {};
  render();
}

function probBars(filename) {
  const fname = filename.split('/').pop();
  const p = probs[fname];
  if (!p) return '';
  const rows = classes.map((cls, i) => {
    const color = CLASS_COLORS[i] || '#888888';
    const val = p[cls] !== undefined ? p[cls] : 0;
    const pct = (val * 100).toFixed(0);
    return `<div style="display:flex;align-items:center;gap:5px;margin:2px 0">
      <span style="flex:0 0 40%;min-width:0;font-size:10px;color:${color};word-break:break-all;line-height:1.3">${cls}</span>
      <div style="flex:1;min-width:30px;height:8px;background:#111120;border-radius:2px;align-self:center">
        <div style="width:${pct}%;height:100%;background:${color}bb;border-radius:2px"></div>
      </div>
      <span style="flex:0 0 38px;font-size:10px;color:#888;text-align:right">${val.toFixed(3)}</span>
    </div>`;
  }).join('');
  return `<div style="border-top:1px solid #2a2a4a;padding-top:6px;margin-top:4px">${rows}</div>`;
}

function render() {
  const labeled = Object.keys(labels).length;
  document.getElementById('stats').textContent =
    `${crops.length} кропов · размечено ${labeled}`;

  document.getElementById('bulk-btns').innerHTML =
    classes.map((cls, i) => {
      const color = CLASS_COLORS[i] || '#888';
      return `<button class="btn-bulk btn-bulk-cls" onclick="applyBulk('${cls}')"
        style="border-left-color:${color}">${cls}</button>`;
    }).join('') +
    `<button class="btn-bulk btn-bulk-skip" onclick="applyBulk('skip')">Пропустить</button>`;

  const groups = {};
  const ORDER = [...classes, 'skip', 'unknown'];
  ORDER.forEach(c => groups[c] = []);
  groups['—'] = [];

  crops.forEach((f, i) => {
    const lbl = labels[f] || '—';
    const key = ORDER.includes(lbl) ? lbl : '—';
    groups[key].push(i);
  });

  const sections = [...ORDER, '—'].filter(k => groups[k] && groups[k].length);

  flatOrder = [];
  sections.forEach(cls => groups[cls].forEach(i => flatOrder.push(i)));

  document.getElementById('gallery').innerHTML = sections.map(cls => {
    const cidx = classes.indexOf(cls);
    const color = cidx >= 0 ? CLASS_COLORS[cidx] : (cls === '—' ? '#444' : '#888');
    const tiles = groups[cls].map(i => {
      const f = crops[i];
      const name = f.split('/').pop();
      const sel = selected.has(i);
      return `<div class="tile${sel ? ' selected' : ''}" data-idx="${i}"
        onclick="tileClick(${i}, event)" title="${name}">
        <div class="tile-check" onclick="toggleSelect(${i}, event)">${sel ? '✓' : ''}</div>
        <img src="/image/${encodeURIComponent(f)}" loading="lazy">
        <div class="tile-label">${name}</div>
      </div>`;
    }).join('');
    return `<div class="section">
      <div class="section-title" style="background:${color}22;color:${color}">
        ${cls === '—' ? 'без метки' : cls} &nbsp;(${groups[cls].length})
      </div>
      <div class="grid">${tiles}</div>
    </div>`;
  }).join('');

  updateBulkBar();
}

function tileClick(i, e) {
  if (e.ctrlKey || e.metaKey || e.shiftKey) toggleSelect(i, e);
  else openModal(i);
}

function toggleSelect(idx, e) {
  e.stopPropagation();
  if (e.shiftKey && lastToggleIdx >= 0) {
    const a = flatOrder.indexOf(lastToggleIdx);
    const b = flatOrder.indexOf(idx);
    if (a >= 0 && b >= 0) {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      for (let j = lo; j <= hi; j++) selected.add(flatOrder[j]);
    } else {
      selected.add(idx);
      lastToggleIdx = idx;
    }
  } else {
    if (selected.has(idx)) selected.delete(idx);
    else { selected.add(idx); lastToggleIdx = idx; }
  }
  updateSelectionDOM();
  updateBulkBar();
}

function selectAll() {
  for (let i = 0; i < crops.length; i++) selected.add(i);
  updateSelectionDOM();
  updateBulkBar();
}

function deselectAll() {
  selected.clear();
  updateSelectionDOM();
  updateBulkBar();
}

function updateSelectionDOM() {
  document.querySelectorAll('.tile[data-idx]').forEach(el => {
    const i = +el.dataset.idx;
    const sel = selected.has(i);
    el.classList.toggle('selected', sel);
    el.querySelector('.tile-check').textContent = sel ? '✓' : '';
  });
}

function _syncGalleryPadding() {
  const bar = document.getElementById('bulk-bar');
  document.getElementById('gallery').style.paddingBottom = bar.offsetHeight + 'px';
}
new ResizeObserver(_syncGalleryPadding).observe(document.getElementById('bulk-bar'));

function updateBulkBar() {
  document.getElementById('bulk-count').textContent = 'Выбрано: ' + selected.size;
  document.getElementById('bulk-bar').classList.toggle('show', selected.size > 0);
  requestAnimationFrame(_syncGalleryPadding);
}

async function applyBulk(cls) {
  if (!selected.size) return;
  const files = [...selected].map(i => crops[i]);
  await fetch('/api/label/bulk', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({files, label: cls})
  });
  const d = await (await fetch('/api/state')).json();
  labels = d.labels;
  selected.clear();
  lastToggleIdx = -1;
  render();
}

function openModal(i) {
  modalIdx = i;
  modalFile = crops[i];
  const lbl = labels[modalFile] || '';
  document.getElementById('modal-img').src = '/image/' + encodeURIComponent(modalFile);
  document.getElementById('modal-fname').textContent = modalFile.split('/').pop();
  document.getElementById('modal-probs').innerHTML = probBars(modalFile);
  const btns = classes.map((cls, ci) => {
    const color = CLASS_COLORS[ci] || '#888';
    const active = lbl === cls;
    return `<button class="btn-class${active?' active':''}" onclick="relabel('${cls}')"
      style="${active?'border-left-color:'+color+';color:'+color:''}">${cls}</button>`;
  }).join('');
  document.getElementById('modal-btns').innerHTML =
    btns +
    `<button class="btn-skip" onclick="relabel('skip')">Пропустить</button>` +
    `<button class="btn-go" onclick="goLabel()">→ Разметка</button>` +
    `<button class="btn-close" onclick="closeModal()">✕ Закрыть</button>`;
  document.getElementById('modal').classList.add('open');
}

async function relabel(cls) {
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({file: modalFile, label: cls})
  });
  const d = await (await fetch('/api/state')).json();
  labels = d.labels;
  lastToggleIdx = -1;
  openModal(modalIdx);
  render();
}

function goLabel() {
  location.href = '/?idx=' + modalIdx;
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

document.getElementById('modal').addEventListener('click', e => {
  if (e.target === document.getElementById('modal')) closeModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

loadState();
</script>
</body>
</html>"""

_DATASET_GALLERY_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Датасет</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: monospace; background: #1a1a2e; color: #eee; margin: 0; }
  h2 { color: #7ec8e3; margin: 0; padding: 8px 16px; font-size: 15px;
       border-bottom: 1px solid #2a2a4a; display: flex; align-items: center; gap: 16px;
       position: sticky; top: 0; background: #1a1a2e; z-index: 10; }
  h2 a { color: #7ec8e3; font-size: 12px; opacity: 0.6; text-decoration: none; }
  h2 a:hover { opacity: 1; }
  h2 .stats { margin-left: auto; font-size: 12px; color: #555; }
  .section { padding: 12px 16px 4px; }
  .section-title { font-size: 13px; font-weight: bold; margin-bottom: 8px;
                   padding: 4px 10px; border-radius: 4px; display: inline-block; }
  :root { --tw: 120px; }
  .grid { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .tile { width: var(--tw); cursor: pointer; border: 2px solid #2a2a4a; border-radius: 4px;
          overflow: hidden; background: #111120; transition: border-color .15s; position: relative; }
  .tile:hover { border-color: #7ec8e3; }
  .tile.selected { border-color: #7ec8e3 !important; background: #0f2040; }
  .tile-check { position: absolute; top: 4px; left: 4px; z-index: 5; width: 20px; height: 20px;
                background: rgba(0,0,0,.65); border: 2px solid #556; border-radius: 3px;
                cursor: pointer; display: flex; align-items: center; justify-content: center;
                font-size: 13px; opacity: 0; transition: opacity .1s, border-color .1s; }
  .tile:hover .tile-check, .tile.selected .tile-check { opacity: 1; }
  .tile.selected .tile-check { border-color: #7ec8e3; color: #7ec8e3; background: rgba(0,30,60,.85); }
  .tile img { width: var(--tw); height: calc(var(--tw) * 0.75); object-fit: contain; display: block; background: #0a0a18; }
  .tile .tile-label { font-size: 9px; color: #777; padding: 3px 4px;
                      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85);
           z-index: 100; align-items: center; justify-content: center; flex-direction: column; gap: 12px; }
  #modal.open { display: flex; }
  #modal img { max-width: 90vw; max-height: 70vh; object-fit: contain; border: 2px solid #444; }
  #modal .modal-cls { font-size: 13px; color: #aaa; }
  #modal .modal-fname { color: #555; font-size: 11px; }
  #modal .modal-probs { width: 60vw; min-width: 300px; max-width: 560px; }
  #modal .modal-btns { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  #modal button { padding: 8px 16px; font-size: 13px; cursor: pointer; border: none;
                  border-radius: 4px; font-family: monospace; }
  #modal .btn-class { background: #16213e; color: #eee; border-left: 4px solid #555; }
  #modal .btn-class.active { border-left-color: #f9ca24; background: #0f3460; color: #f9ca24; }
  #modal .btn-close { background: #2a2a4a; color: #aaa; }
  #modal .btn-skip { background: #2d1b1b; color: #e74c3c; }
  #modal .btn-skip:hover { background: #4a2020; color: #ff6b6b; }
  #modal .moving { opacity: 0.5; pointer-events: none; }
  .zoom-btn { background: #2a2a4a; color: #aaa; border: none; border-radius: 3px;
              cursor: pointer; font-size: 12px; padding: 2px 8px; font-family: monospace; }
  .zoom-btn:hover { background: #3a3a6a; color: #eee; }
  #bulk-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #0d1522;
              border-top: 2px solid #7ec8e3; padding: 8px 16px; display: none;
              align-items: center; gap: 8px; z-index: 50; flex-wrap: wrap; }
  #bulk-bar.show { display: flex; }
  #bulk-count { color: #7ec8e3; font-size: 13px; white-space: nowrap; min-width: 90px; }
  #bulk-btns { display: flex; gap: 6px; flex-wrap: wrap; }
  .btn-bulk { padding: 5px 11px; font-size: 12px; cursor: pointer; border: none;
              border-radius: 4px; font-family: monospace; }
  .btn-bulk-cls { background: #16213e; color: #eee; border-left: 3px solid #555; }
  .btn-bulk-cls:hover { background: #0f3460; color: #f9ca24; }
  .btn-bulk-desel { background: #2a2a4a; color: #999; margin-left: auto; }
  .btn-bulk-desel:hover { color: #eee; }
  #gallery { padding-bottom: 0; }
</style>
</head>
<body>
<h2>Датасет
  <a href="/">← Разметка</a>
  <a href="/gallery">⊞ Галерея</a>
  <span style="display:flex;gap:4px;align-items:center">
    <button class="zoom-btn" onclick="selectAll()">☑ Все</button>
    <button class="zoom-btn" onclick="deselectAll()">☐ Снять</button>
    <button class="zoom-btn" onclick="zoom(-1)">−</button>
    <button class="zoom-btn" onclick="zoom(+1)">+</button>
  </span>
  <span class="stats" id="stats"></span>
</h2>
<div id="gallery"></div>

<div id="bulk-bar">
  <span id="bulk-count">Выбрано: 0</span>
  <div id="bulk-btns"></div>
  <button class="btn-bulk btn-bulk-desel" onclick="deselectAll()">✕ Снять</button>
</div>

<div id="modal">
  <img id="modal-img" src="" alt="">
  <div class="modal-cls" id="modal-cls"></div>
  <div class="modal-fname" id="modal-fname"></div>
  <div class="modal-probs" id="modal-probs"></div>
  <div class="modal-btns" id="modal-btns"></div>
</div>

<script>
const CLASS_COLORS = ["#27ae60","#e67e22","#9b59b6","#95a5a6","#3498db",
                      "#e74c3c","#1abc9c","#f39c12","#8e44ad"];
let groups = {}, classes = [];
let probs = {};
let modalFile = '', modalCls = '';
const ZOOM_STEPS = [60, 90, 120, 180, 240, 360];
let zoomIdx = 2;
let selected = new Set();  // indices into allFiles
let allFiles = [];          // [{f, cls}, ...] flat list built each render
let lastToggleFile = null;  // anchor tracked by file path (allFiles indices shift on render)

function zoom(d) {
  zoomIdx = Math.max(0, Math.min(ZOOM_STEPS.length - 1, zoomIdx + d));
  document.documentElement.style.setProperty('--tw', ZOOM_STEPS[zoomIdx] + 'px');
}

async function loadState() {
  const d = await (await fetch('/api/dataset')).json();
  groups = d.groups; classes = d.classes;
  probs = d.probs || {};
  render();
}

function probBars(filename) {
  const fname = filename.split('/').pop();
  const p = probs[fname];
  if (!p) return '';
  const rows = classes.map((cls, i) => {
    const color = CLASS_COLORS[i] || '#888888';
    const val = p[cls] !== undefined ? p[cls] : 0;
    const pct = (val * 100).toFixed(0);
    return `<div style="display:flex;align-items:center;gap:5px;margin:2px 0">
      <span style="flex:0 0 40%;min-width:0;font-size:10px;color:${color};word-break:break-all;line-height:1.3">${cls}</span>
      <div style="flex:1;min-width:30px;height:8px;background:#111120;border-radius:2px;align-self:center">
        <div style="width:${pct}%;height:100%;background:${color}bb;border-radius:2px"></div>
      </div>
      <span style="flex:0 0 38px;font-size:10px;color:#888;text-align:right">${val.toFixed(3)}</span>
    </div>`;
  }).join('');
  return `<div style="border-top:1px solid #2a2a4a;padding-top:6px;margin-top:4px">${rows}</div>`;
}

function render() {
  const total = Object.values(groups).reduce((s, a) => s + a.length, 0);
  document.getElementById('stats').textContent = `${total} файлов`;

  document.getElementById('bulk-btns').innerHTML = classes.map((cls, i) => {
    const color = CLASS_COLORS[i] || '#888';
    return `<button class="btn-bulk btn-bulk-cls" onclick="applyBulk('${cls}')"
      style="border-left-color:${color}">${cls}</button>`;
  }).join('');

  // rebuild flat list for range selection
  allFiles = [];
  const orderedCls = classes.filter(cls => groups[cls] && groups[cls].length);
  orderedCls.forEach(cls => groups[cls].forEach(f => allFiles.push({f, cls})));

  let fi = 0;
  document.getElementById('gallery').innerHTML = orderedCls.map(cls => {
    const cidx = classes.indexOf(cls);
    const color = CLASS_COLORS[cidx] || '#888';
    const tiles = groups[cls].map(f => {
      const i = fi++;
      const sel = selected.has(i);
      const name = f.split('/').pop();
      return `<div class="tile${sel ? ' selected' : ''}" data-fidx="${i}"
        onclick="tileClick(${i},'${f}','${cls}',event)" title="${name}">
        <div class="tile-check" onclick="toggleSelect(${i},event)">${sel ? '✓' : ''}</div>
        <img src="/image/${encodeURIComponent(f)}" loading="lazy">
        <div class="tile-label">${name}</div>
      </div>`;
    }).join('');
    return `<div class="section">
      <div class="section-title" style="background:${color}22;color:${color}">
        ${cls} &nbsp;(${groups[cls].length})
      </div>
      <div class="grid">${tiles}</div>
    </div>`;
  }).join('');

  updateBulkBar();
}

function tileClick(i, f, cls, e) {
  if (e.ctrlKey || e.metaKey || e.shiftKey) toggleSelect(i, e);
  else openModal(f, cls);
}

function toggleSelect(idx, e) {
  e.stopPropagation();
  if (e.shiftKey && lastToggleFile !== null) {
    const anchorPos = allFiles.findIndex(x => x.f === lastToggleFile);
    if (anchorPos >= 0) {
      const lo = Math.min(anchorPos, idx), hi = Math.max(anchorPos, idx);
      for (let j = lo; j <= hi; j++) selected.add(j);
    } else {
      selected.add(idx);
      lastToggleFile = allFiles[idx].f;
    }
  } else {
    if (selected.has(idx)) selected.delete(idx);
    else { selected.add(idx); lastToggleFile = allFiles[idx].f; }
  }
  updateSelectionDOM();
  updateBulkBar();
}

function selectAll() {
  for (let i = 0; i < allFiles.length; i++) selected.add(i);
  updateSelectionDOM();
  updateBulkBar();
}

function deselectAll() {
  selected.clear();
  updateSelectionDOM();
  updateBulkBar();
}

function updateSelectionDOM() {
  document.querySelectorAll('.tile[data-fidx]').forEach(el => {
    const i = +el.dataset.fidx;
    const sel = selected.has(i);
    el.classList.toggle('selected', sel);
    el.querySelector('.tile-check').textContent = sel ? '✓' : '';
  });
}

function _syncGalleryPadding() {
  const bar = document.getElementById('bulk-bar');
  document.getElementById('gallery').style.paddingBottom = bar.offsetHeight + 'px';
}
new ResizeObserver(_syncGalleryPadding).observe(document.getElementById('bulk-bar'));

function updateBulkBar() {
  document.getElementById('bulk-count').textContent = 'Выбрано: ' + selected.size;
  document.getElementById('bulk-bar').classList.toggle('show', selected.size > 0);
  requestAnimationFrame(_syncGalleryPadding);
}

async function applyBulk(toCls) {
  if (!selected.size) return;
  const files = [...selected].map(i => allFiles[i].f);
  await fetch('/api/dataset/move/bulk', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({files, to_class: toCls})
  });
  const d = await (await fetch('/api/dataset')).json();
  groups = d.groups; classes = d.classes;
  selected.clear();
  lastToggleFile = null;
  render();
}

function openModal(f, cls) {
  modalFile = f; modalCls = cls;
  document.getElementById('modal-img').src = '/image/' + encodeURIComponent(f);
  document.getElementById('modal-cls').textContent = cls;
  document.getElementById('modal-fname').textContent = f.split('/').pop();
  document.getElementById('modal-probs').innerHTML = probBars(f);
  buildModalBtns();
  document.getElementById('modal').classList.add('open');
}

function buildModalBtns() {
  const btns = classes.map((cls, i) => {
    const color = CLASS_COLORS[i] || '#888';
    const active = cls === modalCls;
    return `<button class="btn-class${active?' active':''}" onclick="moveTo('${cls}')"
      style="${active?'border-left-color:'+color+';color:'+color:''}">${cls}</button>`;
  }).join('');
  document.getElementById('modal-btns').innerHTML =
    btns +
    `<button class="btn-skip" onclick="moveTo('skip')">Пропустить</button>` +
    `<button class="btn-close" onclick="closeModal()">✕ Закрыть</button>`;
}

async function moveTo(toCls) {
  if (toCls === modalCls) { closeModal(); return; }
  document.getElementById('modal-btns').classList.add('moving');
  const r = await fetch('/api/dataset/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({file: modalFile, to_class: toCls})
  });
  const d = await r.json();
  if (d.ok) {
    groups[modalCls] = groups[modalCls].filter(f => f !== modalFile);
    if (!groups[toCls]) groups[toCls] = [];
    groups[toCls].push(d.new_path);
    modalFile = d.new_path;
    modalCls = toCls;
    lastToggleFile = null;
    render();
    buildModalBtns();
    document.getElementById('modal-cls').textContent = toCls;
  }
  document.getElementById('modal-btns').classList.remove('moving');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

document.getElementById('modal').addEventListener('click', e => {
  if (e.target === document.getElementById('modal')) closeModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

loadState();
</script>
</body>
</html>"""


def _load_probs(csv_path: Path, classes: list[str]) -> dict[str, dict[str, float]]:
    """Читает p_<class> колонки из classifications.csv → {filename: {class: prob}}."""
    import csv as _csv
    if not csv_path.is_file():
        return {}
    result: dict[str, dict[str, float]] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                crop = row.get("crop", "")
                if not crop:
                    continue
                cls_probs: dict[str, float] = {}
                for cls in classes:
                    raw = row.get(f"p_{cls}", "")
                    try:
                        cls_probs[cls] = float(raw) if raw else 0.0
                    except ValueError:
                        cls_probs[cls] = 0.0
                if any(v > 0 for v in cls_probs.values()):
                    result[crop] = cls_probs
    except Exception:
        pass
    return result


def run_server(input_dir: Path, port: int, labels_path: Path,
               unlabeled_only: bool = False,
               classes: list[str] | None = None,
               dataset_dir: Path | None = None,
               image_exts: set[str] | None = None,
               allowed_ips: IpAllowlist | None = None,
               probs: dict | None = None) -> None:
    from flask import Flask, abort, jsonify, request, send_file, Response
    import logging
    import shutil as _shutil

    log = logging.getLogger(__name__)
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.before_request
    def _check_client_ip():
        if not is_ip_allowed(request.remote_addr, allowed_ips):
            log_ip_denied(
                log,
                client_ip=request.remote_addr,
                service="Label UI",
                path=request.path,
            )
            abort(403)

    _exts = image_exts if image_exts is not None else IMAGE_EXTS

    # Собираем картинки рекурсивно из input_dir, фильтруя по расширению
    crops: list[str] = []
    for f in sorted(input_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in _exts:
            crops.append(f.resolve().as_posix())

    # Загружаем существующую разметку
    labels: dict[str, str] = {}
    if labels_path.is_file():
        existing = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = existing.get("labels", existing)  # плоский или вложенный формат

    # Нормализуем ключи labels к абсолютным путям кропов.
    # labels.json может хранить: абсолютный путь, basename, или basename с префиксом scene0042_.
    # Сопоставляем с реальными файлами в crops по stripped-basename.
    _re_scene = re.compile(r"^scene\d+_")
    _stripped_to_crop = {_re_scene.sub("", Path(c).name): c for c in crops}
    for k, v in list(labels.items()):
        stripped = _re_scene.sub("", Path(k).name)
        crop_path = _stripped_to_crop.get(stripped)
        if crop_path and crop_path not in labels:
            labels[crop_path] = v

    _classes = classes or []
    _valid_set = set(_classes)
    if unlabeled_only:
        crops = [c for c in crops if c not in labels or labels[c] not in _valid_set]

    current = 0

    @app.route("/")
    def index():
        return Response(_HTML, mimetype="text/html")

    @app.route("/gallery")
    def gallery():
        return Response(_GALLERY_HTML, mimetype="text/html")

    @app.route("/gallery/dataset")
    def gallery_dataset():
        return Response(_DATASET_GALLERY_HTML, mimetype="text/html")

    _probs = probs or {}

    @app.route("/api/state")
    def state():
        return jsonify({"crops": crops, "labels": labels, "current_idx": current,
                        "classes": _classes, "probs": _probs})

    @app.route("/api/dataset")
    def api_dataset():
        if dataset_dir is None or not dataset_dir.exists():
            return jsonify({"error": "no dataset"}), 404
        grps: dict[str, list[str]] = {}
        for subdir in sorted(dataset_dir.iterdir()):
            if not subdir.is_dir():
                continue
            files = sorted(
                f.resolve().as_posix()
                for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
            if files:
                grps[subdir.name] = files
        return jsonify({"groups": grps, "classes": _classes, "probs": _probs})

    @app.route("/api/dataset/move", methods=["POST"])
    def api_dataset_move():
        if dataset_dir is None:
            return jsonify({"error": "no dataset"}), 404
        data = request.get_json()
        src = REPO_ROOT / data["file"]
        to_cls = data["to_class"]
        dst_dir = dataset_dir / to_cls
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / src.name
        if src.exists() and not dst.exists():
            _shutil.move(str(src), dst)
        return jsonify({"ok": True, "new_path": dst.resolve().as_posix()})

    @app.route("/api/dataset/move/bulk", methods=["POST"])
    def api_dataset_move_bulk():
        if dataset_dir is None:
            return jsonify({"error": "no dataset"}), 404
        data = request.get_json()
        files = data.get("files", [])
        to_cls = data.get("to_class", "")
        if not to_cls or not files:
            return jsonify({"error": "missing params"}), 400
        dst_dir = dataset_dir / to_cls
        dst_dir.mkdir(exist_ok=True)
        moved = []
        for fpath in files:
            src = Path(fpath) if Path(fpath).is_absolute() else REPO_ROOT / fpath
            if src.exists():
                dst = dst_dir / src.name
                if not dst.exists():
                    _shutil.move(str(src), dst)
                moved.append(dst.resolve().as_posix())
        return jsonify({"ok": True, "moved": len(moved)})

    @app.route("/api/label", methods=["POST"])
    def label():
        nonlocal current
        data = request.get_json()
        fname, cls = data["file"], data["label"]
        if cls is not None:
            labels[fname] = cls
        _save_labels(labels_path, labels)
        return jsonify({"labels": labels})

    @app.route("/api/label/bulk", methods=["POST"])
    def label_bulk():
        data = request.get_json()
        files = data.get("files", [])
        cls = data.get("label")
        if cls and files:
            crop_set = set(crops)
            for f in files:
                if f in crop_set:
                    labels[f] = cls
            _save_labels(labels_path, labels)
        return jsonify({"labels": labels, "updated": len(files)})

    @app.route("/image/<path:fname>")
    def image(fname: str):
        # Flask strips the leading '/' from path params; restore it for absolute paths
        absolute = Path("/" + fname)
        full = absolute if absolute.is_file() else REPO_ROOT / fname
        if not full.is_file():
            return Response("not found", status=404)
        mime = mimetypes.guess_type(str(full))[0] or "image/jpeg"
        return send_file(str(full), mimetype=mime)

    url = f"http://127.0.0.1:{port}"
    print(f"Разметчик запущен: {url}")
    print(f"  {url}/gallery         — галерея сессии")
    if dataset_dir:
        print(f"  {url}/gallery/dataset — галерея датасета ({dataset_dir.name})")
    mode = "только неразмеченные" if unlabeled_only else "все"
    print(f"Кропов: {len(crops)} ({mode})  Уже размечено: {len(labels)}")
    print(f"Метки: {labels_path}")
    if allowed_ips is not None:
        print(f"Allowed IPs: {allowed_ips.summary()}")
    else:
        print("[!] ALLOWED_IPS не задан — доступ с любого IP", file=sys.stderr)
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
    parser.add_argument("--input",   type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels",  type=Path, default=None,
                        help="Файл для сохранения меток (default: .output/train/2_label_ui/labels.json)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Каталог датасета для чтения классов (default: .data/groups/последняя версия)")
    parser.add_argument("--port",           type=int, default=_DEFAULT_PORT)
    parser.add_argument("--unlabeled-only", action="store_true",
                        help="Показывать только ещё не размеченные кропы")
    parser.add_argument("--ext", default="jpg",
                        help="Расширения файлов через запятую (default: jpg)")
    parser.add_argument("--probs", action="store_true",
                        help="Показывать вероятности из classifications.csv")
    parser.add_argument("--probs-csv", type=Path, default=None,
                        help="Путь к CSV с вероятностями (переопределяет --probs)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Нет каталога: {args.input}", file=sys.stderr)
        return 1

    classes, resolved_dataset = _load_classes(args.dataset)
    print(f"Классы ({len(classes)}): {', '.join(classes)}")

    labels_path = args.labels or (DEFAULT_LABELS / "labels.json")
    image_exts = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    allowed_ips = parse_allowed_ips(os.environ.get("ALLOWED_IPS", ""))

    probs: dict | None = None
    if args.probs_csv:
        csv_path = args.probs_csv
        probs = _load_probs(csv_path, classes)
        print(f"Вероятности: {csv_path}  ({len(probs)} записей)")
    elif args.probs:
        csv_path = labels_path.parent / "classifications.csv"
        probs = _load_probs(csv_path, classes)
        print(f"Вероятности: {csv_path}  ({len(probs)} записей)")

    run_server(args.input, args.port, labels_path,
               unlabeled_only=args.unlabeled_only,
               classes=classes,
               dataset_dir=resolved_dataset,
               image_exts=image_exts,
               allowed_ips=allowed_ips,
               probs=probs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
