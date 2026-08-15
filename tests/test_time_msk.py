"""Тесты часового пояса меток (time_msk): дефолт МСК + настройка TZ_OFFSET_HOURS."""
from __future__ import annotations

from common.utils import time_msk


def test_default_msk_offset():
    # epoch 0 = 1970-01-01 00:00 UTC → МСК (+3) = 03:00
    time_msk.set_tz_offset(3)
    assert time_msk.ts_file_from_epoch(0).startswith("19700101_030000")


def test_tz_offset_configurable():
    time_msk.set_tz_offset(5)
    assert time_msk.ts_file_from_epoch(0).startswith("19700101_050000")
    time_msk.set_tz_offset(0)
    assert time_msk.ts_file_from_epoch(0).startswith("19700101_000000")
    time_msk.set_tz_offset(3)   # вернуть дефолт для остальных тестов


def test_set_tz_offset_from_env(monkeypatch):
    monkeypatch.setenv("TZ_OFFSET_HOURS", "7")
    time_msk.set_tz_offset()    # None → перечитать из окружения
    assert time_msk.ts_file_from_epoch(0).startswith("19700101_070000")
    monkeypatch.delenv("TZ_OFFSET_HOURS", raising=False)
    time_msk.set_tz_offset(3)   # вернуть дефолт


def test_suffix_always_msk():
    # суффикс имён всегда '_msk' независимо от смещения
    time_msk.set_tz_offset(0)
    assert time_msk.ts_file_from_epoch(0).endswith("_msk")
    time_msk.set_tz_offset(3)
