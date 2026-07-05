"""Управление версиями обученных ONNX-моделей классификатора.

Используется как бэкендом (get_active_model_path), так и CLI-скриптами.

Версии хранятся в .models/classify/:
  v1_1.onnx, v1_1.json  — 1-й прогон на датасете .data/groups/v1
  v1_2.onnx, v1_2.json  — 2-й прогон на том же датасете
  v3.onnx, v3.json      — legacy: без привязки к датасету (export.zip и т.п.)

JSON-манифест:
  {"tag": "v1_2", "dataset_version": "v1", "dataset_path": ".data/groups/v1", ...}
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from common.utils.atomic import copy as _copy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / ".models" / "classify"
_CONFIG_YAML = _REPO_ROOT / "config.yaml"

_DATASET_DIR_RE = re.compile(r"^v\d+$")
_TAG_DATASET_RE = re.compile(r"^(v\d+)_(\d+)$")
_TAG_LEGACY_RE = re.compile(r"^v(\d+)$")


# ── Вспомогательные пути ──────────────────────────────────────────────────────

def models_dir() -> Path:
    return _MODELS_DIR


def classify_model_path(tag: str) -> Path:
    return _MODELS_DIR / f"{tag}.onnx"


def classify_manifest_path(tag: str) -> Path:
    return _MODELS_DIR / f"{tag}.json"


def model_path(version: int) -> Path:
    """Legacy: целочисленная версия v{N}.onnx."""
    return classify_model_path(f"v{version}")


def manifest_path(version: int) -> Path:
    return classify_manifest_path(f"v{version}")


# ── Датасет → тег модели ──────────────────────────────────────────────────────

def resolve_dataset_version(data_path: Path) -> tuple[str | None, Path | None]:
    """Извлекает v1 из .data/groups/v1 или dataset.json. Возвращает (version, dataset_dir)."""
    p = data_path.resolve()
    if _DATASET_DIR_RE.match(p.name):
        return p.name, p

    for candidate in (p, p.parent):
        meta = candidate / "dataset.json"
        if meta.is_file():
            try:
                raw = json.loads(meta.read_text(encoding="utf-8"))
                ver = raw.get("version")
                if isinstance(ver, int):
                    return f"v{ver}", candidate
                if isinstance(ver, str) and _DATASET_DIR_RE.match(ver):
                    return ver, candidate
            except Exception:
                pass
        if _DATASET_DIR_RE.match(candidate.name):
            return candidate.name, candidate

    return None, None


def next_classify_model_tag(dataset_version: str | None) -> str:
    """Следующий тег модели: v1_3 для датасета v1 или legacy v{N}."""
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)

    if dataset_version:
        runs: list[int] = []
        for f in d.glob(f"{dataset_version}_*.onnx"):
            m = _TAG_DATASET_RE.match(f.stem)
            if m and m.group(1) == dataset_version:
                runs.append(int(m.group(2)))
        return f"{dataset_version}_{max(runs, default=0) + 1}"

    nums: list[int] = []
    for f in d.glob("v*.onnx"):
        m = _TAG_LEGACY_RE.match(f.stem)
        if m:
            nums.append(int(m.group(1)))
    return f"v{max(nums, default=0) + 1}"


def write_classify_manifest(
    tag: str,
    *,
    metrics: dict | None = None,
    dataset_version: str | None = None,
    dataset_path: str = "",
    notes: str = "",
) -> None:
    classify_manifest_path(tag).write_text(
        json.dumps({
            "tag": tag,
            "dataset_version": dataset_version,
            "dataset_path": dataset_path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
            "metrics": metrics or {},
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Чтение состояния ──────────────────────────────────────────────────────────

def _parse_tag(tag: str) -> dict:
    info: dict = {"tag": tag}
    m = _TAG_DATASET_RE.match(tag)
    if m:
        info["dataset_version"] = m.group(1)
        info["run"] = int(m.group(2))
        info["version"] = info["run"]
        return info
    m = _TAG_LEGACY_RE.match(tag)
    if m:
        info["version"] = int(m.group(1))
        return info
    return info


def list_versions() -> list[dict]:
    """Все ONNX-модели в .models/classify/, отсортированные по тегу."""
    versions: list[dict] = []
    if not _MODELS_DIR.is_dir():
        return versions
    for f in sorted(_MODELS_DIR.glob("*.onnx")):
        tag = f.stem
        info: dict = {"tag": tag, "file": f.name, "size_kb": f.stat().st_size // 1024}
        info.update(_parse_tag(tag))
        mp = classify_manifest_path(tag)
        if mp.is_file():
            try:
                info.update(json.loads(mp.read_text(encoding="utf-8")))
            except Exception:
                pass
        versions.append(info)
    return versions


def get_active_tag() -> Optional[str]:
    """Тег активной модели (stem файла из config.yaml)."""
    p = get_active_model_path()
    return p.stem if p else None


def get_active_version() -> Optional[int]:
    """Legacy: номер v{N} или run из v1_N."""
    tag = get_active_tag()
    if not tag:
        return None
    info = _parse_tag(tag)
    return info.get("version")


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

def activate_tag(tag: str) -> bool:
    """Устанавливает модель по тегу (v1_2, v3, …) как активную в config.yaml."""
    mp = classify_model_path(tag)
    if not mp.is_file() or not _CONFIG_YAML.is_file():
        return False
    try:
        text = _CONFIG_YAML.read_text(encoding="utf-8")
        new_path = str(mp.relative_to(_REPO_ROOT)).replace("\\", "/")
        new_text = re.sub(r"(classify\s*:\s*).*", lambda m: m.group(1) + new_path, text)
        _CONFIG_YAML.write_text(new_text, encoding="utf-8")
        return True
    except Exception:
        return False


def activate_version(version: int) -> bool:
    """Legacy: активировать v{N}."""
    return activate_tag(f"v{version}")


def register_version(onnx_src: Path, metrics: dict | None = None, notes: str = "") -> str:
    """Копирует ONNX в .models/classify/, создаёт манифест. Возвращает тег."""
    tag = next_classify_model_tag(None)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _copy(onnx_src, classify_model_path(tag))
    write_classify_manifest(tag, metrics=metrics, notes=notes, dataset_path=str(onnx_src))
    return tag


def clean_versions(keep: int) -> list[str]:
    """Удаляет старые версии, оставляет `keep` последних + активную."""
    active = get_active_tag()
    versions = list_versions()
    to_keep = {v["tag"] for v in versions[-keep:]}
    if active:
        to_keep.add(active)
    removed: list[str] = []
    for v in versions:
        tag = v["tag"]
        if tag not in to_keep:
            classify_model_path(tag).unlink(missing_ok=True)
            classify_manifest_path(tag).unlink(missing_ok=True)
            removed.append(tag)
    return removed
