"""Тесты версии проекта: файл VERSION, helper get_version, логика bump."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from common.version import get_version

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import version as version_cli  # noqa: E402


def test_version_file_exists():
    assert (REPO_ROOT / "VERSION").is_file()


def test_version_format():
    v = get_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), f"неверный формат версии: {v}"


def test_get_version_matches_file():
    on_disk = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert get_version() == on_disk


def test_bump_minor():
    assert version_cli.next_version("0.1.2", "minor") == "0.2.0"


def test_bump_patch():
    assert version_cli.next_version("0.1.2", "patch") == "0.1.3"


def test_bump_major():
    assert version_cli.next_version("0.1.2", "major") == "1.0.0"


def test_parse_short_form():
    assert version_cli.parse("1.4") == (1, 4, 0)
