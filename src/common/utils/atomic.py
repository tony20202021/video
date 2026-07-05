"""Атомарные операции записи файлов.

Запись через .tmp + rename гарантирует, что читатель видит файл либо
полностью записанным, либо не видит его вовсе — никаких частичных файлов.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def imwrite(path: Path | str, img) -> bool:
    """Атомарная запись изображения: cv2.imwrite → .tmp.<ext> → rename.

    Возвращает True при успехе (как cv2.imwrite).
    При ошибке кодирования .tmp файл удаляется, dst не трогается.
    """
    import cv2
    path = Path(path)
    # Сохраняем расширение чтобы cv2 знал формат (.tmp.jpg, не .tmp)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    ok = cv2.imwrite(str(tmp), img)
    if ok:
        tmp.rename(path)
    else:
        tmp.unlink(missing_ok=True)
    return ok


def copy(src: Path | str, dst: Path | str) -> None:
    """Атомарная копия файла: shutil.copy2 → .tmp.<ext> → rename."""
    dst = Path(dst)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    shutil.copy2(src, tmp)
    tmp.rename(dst)
