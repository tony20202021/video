"""Тесты атомарных операций записи файлов (src/common/utils/atomic.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.utils.atomic import copy as _copy
from common.utils.atomic import imwrite as _imwrite


# ─── imwrite ─────────────────────────────────────────────────────────────────

def _black_jpg(h: int = 64, w: int = 64) -> object:
    """Возвращает чёрный BGR-кадр для cv2."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_imwrite_creates_file(tmp_path: Path):
    dst = tmp_path / "frame.jpg"
    ok = _imwrite(dst, _black_jpg())
    assert ok is True
    assert dst.is_file()
    assert dst.stat().st_size > 0


def test_imwrite_no_tmp_left(tmp_path: Path):
    """После успешной записи .tmp-файл не остаётся."""
    dst = tmp_path / "frame.jpg"
    _imwrite(dst, _black_jpg())
    tmp = dst.with_name("frame.tmp.jpg")
    assert not tmp.exists()


def test_imwrite_accepts_string_path(tmp_path: Path):
    dst = tmp_path / "frame.jpg"
    ok = _imwrite(str(dst), _black_jpg())
    assert ok is True
    assert dst.is_file()


def test_imwrite_content_is_valid_jpeg(tmp_path: Path):
    import cv2
    dst = tmp_path / "frame.jpg"
    img = _black_jpg(32, 32)
    _imwrite(dst, img)
    loaded = cv2.imread(str(dst))
    assert loaded is not None
    assert loaded.shape == img.shape


def test_imwrite_overwrite_existing(tmp_path: Path):
    """Повторная запись перезаписывает файл."""
    dst = tmp_path / "frame.jpg"
    _imwrite(dst, _black_jpg(32, 32))
    size_before = dst.stat().st_size
    _imwrite(dst, _black_jpg(64, 64))
    assert dst.stat().st_size != size_before


def test_imwrite_failure_returns_false_no_tmp(tmp_path: Path):
    """Если cv2.imwrite провалился — возвращает False, .tmp удалён, dst не создан."""
    dst = tmp_path / "frame.jpg"
    tmp = dst.with_name("frame.tmp.jpg")

    import cv2
    with patch.object(cv2, "imwrite", return_value=False):
        ok = _imwrite(dst, _black_jpg())

    assert ok is False
    assert not dst.exists()
    assert not tmp.exists()


def test_imwrite_failure_leaves_existing_dst_untouched(tmp_path: Path):
    """При ошибке кодирования существующий dst-файл не повреждается."""
    dst = tmp_path / "frame.jpg"
    original = b"original_content"
    dst.write_bytes(original)

    import cv2
    with patch.object(cv2, "imwrite", return_value=False):
        _imwrite(dst, _black_jpg())

    assert dst.read_bytes() == original


# ─── copy ─────────────────────────────────────────────────────────────────────

def test_copy_creates_file(tmp_path: Path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    dst = tmp_path / "sub" / "dst.jpg"
    dst.parent.mkdir()
    _copy(src, dst)
    assert dst.is_file()


def test_copy_no_tmp_left(tmp_path: Path):
    """После успешной копии .tmp-файл не остаётся."""
    src = tmp_path / "src.jpg"
    src.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
    dst = tmp_path / "dst.jpg"
    _copy(src, dst)
    assert not (tmp_path / "dst.tmp.jpg").exists()


def test_copy_content_identical(tmp_path: Path):
    src = tmp_path / "src.bin"
    data = bytes(range(256)) * 10
    src.write_bytes(data)
    dst = tmp_path / "dst.bin"
    _copy(src, dst)
    assert dst.read_bytes() == data


def test_copy_accepts_string_paths(tmp_path: Path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"hello")
    dst = tmp_path / "dst.jpg"
    _copy(str(src), str(dst))
    assert dst.read_bytes() == b"hello"


def test_copy_overwrites_existing_dst(tmp_path: Path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"new_content")
    dst = tmp_path / "dst.jpg"
    dst.write_bytes(b"old_content")
    _copy(src, dst)
    assert dst.read_bytes() == b"new_content"


def test_copy_raises_on_missing_src(tmp_path: Path):
    src = tmp_path / "nonexistent.jpg"
    dst = tmp_path / "dst.jpg"
    with pytest.raises(FileNotFoundError):
        _copy(src, dst)
    assert not (tmp_path / "dst.tmp.jpg").exists()
