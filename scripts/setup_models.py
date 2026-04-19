"""
Подготовка ML-моделей для motion_people.py.

Проверяет наличие models/yolov8n.onnx.
Если файл уже есть — ничего не делает.
Если нет — скачивает yolov8n.pt через ultralytics и экспортирует в ONNX.

Usage:
    python scripts/setup_models.py
    python scripts/setup_models.py --model yolov8n          # другая версия (yolov8s, yolov8m, …)
    python scripts/setup_models.py --force                  # перезаписать если файл есть
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def export_onnx(model_name: str, out_path: Path) -> bool:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Нужен пакет ultralytics: pip install ultralytics", file=sys.stderr)
        return False

    print(f"Скачиваю и экспортирую {model_name} → ONNX …")
    model = YOLO(f"{model_name}.pt")  # скачивает .pt если нет в кэше
    export_result = model.export(format="onnx", imgsz=640)
    # ultralytics возвращает путь к созданному файлу
    src = Path(str(export_result))
    if not src.is_file():
        # fallback: ищем рядом с .pt или в текущем каталоге
        candidates = [
            Path(f"{model_name}.onnx"),
            REPO_ROOT / f"{model_name}.onnx",
            Path.cwd() / f"{model_name}.onnx",
        ]
        src = next((p for p in candidates if p.is_file()), None)
    if src is None or not src.is_file():
        print("Не удалось найти экспортированный .onnx файл.", file=sys.stderr)
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(out_path))
    print(f"Готово: {out_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Подготовка ONNX-моделей")
    ap.add_argument("--model", default="yolov8n", help="Имя модели без расширения (default: yolov8n)")
    ap.add_argument("--force", action="store_true", help="Перезаписать если файл уже существует")
    args = ap.parse_args()

    out_path = MODELS_DIR / f"{args.model}.onnx"

    if out_path.is_file() and not args.force:
        print(f"Модель уже есть: {out_path}")
        return 0

    if args.force and out_path.is_file():
        print(f"--force: перезаписываю {out_path}")

    ok = export_onnx(args.model, out_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
