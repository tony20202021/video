"""Вспомогательные функции для времени в именах файлов. По умолчанию МСК (UTC+3);
смещение настраивается через .env: TZ_OFFSET_HOURS (напр. 3). Суффикс имён — всегда '_msk'."""

import os
from datetime import datetime, timedelta, timezone


def _offset_from_env() -> float:
    try:
        return float(os.environ.get("TZ_OFFSET_HOURS", "3"))
    except (ValueError, TypeError):
        return 3.0


# Часовой пояс меток. Серверные .sh сорсят .env в окружение → подхватывается при импорте.
# motion_diff грузит .env позже (load_dotenv) → вызывает set_tz_offset() после загрузки (см. main).
MSK = timezone(timedelta(hours=_offset_from_env()))


def set_tz_offset(hours: "float | None" = None) -> None:
    """Переустановить пояс меток (после load_dotenv). None → перечитать TZ_OFFSET_HOURS из окружения."""
    global MSK
    MSK = timezone(timedelta(hours=_offset_from_env() if hours is None else float(hours)))


def now_msk() -> datetime:
    return datetime.now(MSK)


def ts_for_file() -> str:
    """Метка времени для имён файлов: YYYYMMDD_HHMMSS_ffffff_msk"""
    return now_msk().strftime("%Y%m%d_%H%M%S_%f_msk")


def ts_for_dir() -> str:
    """Метка времени для имён каталогов: YYYYMMDD_HHMMSS_msk"""
    return now_msk().strftime("%Y%m%d_%H%M%S_msk")


def ts_iso() -> str:
    """ISO 8601 с явным офсетом +03:00 для JSON-полей."""
    return now_msk().isoformat()


def ts_file_from_epoch(epoch: float) -> str:
    """Метка времени для имён файлов из Unix-эпохи (сек) в MSK: YYYYMMDD_HHMMSS_ffffff_msk.
    Используется для штампа по PTS-времени СЪЁМКИ кадра (а не времени обработки)."""
    return datetime.fromtimestamp(epoch, MSK).strftime("%Y%m%d_%H%M%S_%f_msk")


def ts_cam_for_file(cam_dt: "datetime | None") -> str:
    """
    Метка времени камеры для имён файлов: cam_YYYYMMDD_HHMMSS
    Если cam_dt=None — возвращает 'cam_unknown'.
    """
    if cam_dt is None:
        return "cam_unknown"
    return cam_dt.strftime("cam_%Y%m%d_%H%M%S")
