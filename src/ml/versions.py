"""Управление версиями обученных ONNX-моделей классификатора.

Используется как бэкендом (get_active_model_path), так и CLI-скриптами.

Версии хранятся в models/classify/:
  v0.onnx, v0.json  — базовая (pre-trained backbone, случайная голова)
  v1.onnx, v1.json  — первая обученная версия
  ...

JSON-манифест: {"version": N, "created_at": ISO, "notes": "...", "metrics": {...}}
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

# Корень проекта определяется относительно этого файла: src/ml/ → ../.. → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / ".models" / "classify"
_CONFIG_YAML = _REPO_ROOT / "config.yaml"


# ── Вспомогательные пути ──────────────────────────────────────────────────────

def models_dir() -> Path:
    return _MODELS_DIR


def model_path(version: int) -> Path:
    return _MODELS_DIR / f"v{version}.onnx"


def manifest_path(version: int) -> Path:
    return _MODELS_DIR / f"v{version}.json"


# ── Чтение состояния ──────────────────────────────────────────────────────────

def list_versions() -> list[dict]:
    """Возвращает список всех версий (dict) с метриками, отсортированный по номеру."""
    versions = []
    for f in sorted(_MODELS_DIR.glob("v*.onnx")):
        stem = f.stem
        if not stem[1:].isdigit():
            continue
        num = int(stem[1:])
        info: dict = {"version": num, "file": f.name, "size_kb": f.stat().st_size // 1024}
        mp = manifest_path(num)
        if mp.is_file():
            try:
                info.update(json.loads(mp.read_text(encoding="utf-8")))
            except Exception:
                pass
        versions.append(info)
    return versions


def get_active_version() -> Optional[int]:
    """Номер активной версии из config.yaml или None."""
    if not _CONFIG_YAML.is_file():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8"))
        path = cfg.get("models", {}).get("classify", "")
        m = re.search(r"v(\d+)\.onnx", str(path))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def get_active_model_path() -> Optional[Path]:
    """Путь к активной модели из config.yaml. None если не настроено или файл не найден."""
    if not _CONFIG_YAML.is_file():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8"))
        raw = cfg.get("models", {}).get("classify")
        if not raw:
            return None
        p = Path(raw) if Path(raw).is_absolute() else _REPO_ROOT / raw
        return p if p.is_file() else None
    except Exception:
        return None


# ── Изменение состояния ───────────────────────────────────────────────────────

def activate_version(version: int) -> bool:
    """Устанавливает v{version} как активную в config.yaml. Возвращает True при успехе."""
    mp = model_path(version)
    if not mp.is_file():
        return False
    if not _CONFIG_YAML.is_file():
        return False
    try:
        import yaml
        text = _CONFIG_YAML.read_text(encoding="utf-8")
        new_path = str(mp.relative_to(_REPO_ROOT)).replace("\\", "/")
        new_text = re.sub(r"(classify\s*:\s*).*", lambda m: m.group(1) + new_path, text)
        _CONFIG_YAML.write_text(new_text, encoding="utf-8")
        return True
    except Exception:
        return False


def register_version(onnx_src: Path, metrics: dict | None = None, notes: str = "") -> int:
    """Копирует ONNX в models/classify/v{N}.onnx, создаёт манифест. Возвращает номер версии."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_versions()
    next_v = (max(v["version"] for v in existing) + 1) if existing else 0
    shutil.copy2(onnx_src, model_path(next_v))
    manifest_path(next_v).write_text(
        json.dumps({
            "version": next_v,
            "created_at": datetime.now().isoformat(),
            "source": str(onnx_src),
            "notes": notes,
            "metrics": metrics or {},
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return next_v


def clean_versions(keep: int) -> list[int]:
    """Удаляет старые версии, оставляет `keep` последних + активную."""
    active = get_active_version()
    versions = list_versions()
    to_keep = set(v["version"] for v in versions[-keep:])
    if active is not None:
        to_keep.add(active)
    removed = []
    for v in versions:
        num = v["version"]
        if num not in to_keep:
            model_path(num).unlink(missing_ok=True)
            manifest_path(num).unlink(missing_ok=True)
            removed.append(num)
    return removed
