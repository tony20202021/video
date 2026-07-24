"""
4b — темпоральное сглаживание идентификации жителей (Модель 2). Тонкая обёртка над
common.utils.smooth_core (то же ядро, что у 3b для групп).

Читает residents/<ver>/inference/images/<date>/identifications.csv (per-person вероятности p_<person>),
сглаживает person_id по времени внутри визита (в проходе — один человек → идентичность постоянна),
пишет сайдкар residents/<ver>/inference/smoothed/<date>/ (classifications_smoothed.csv crop→person,
smooth_state.json, smooth_viz/). images/ НЕ мутируется.

Набор классов (жителей) открытый → выводится из p_* колонок CSV; подписи/цвета — авто.

Usage:
    python scripts/pipeline/4b_smooth_identity.py .data/residents/v1/inference/images --once
    python scripts/pipeline/4b_smooth_identity.py <root> --prob-window 0 --gate 0 --p-stay 0.9 --viz
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.smooth_core import SmoothCfg, run_service, smooth_date as _core_smooth_date

try:
    _VER = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    _VER = ""

IDENTITY = SmoothCfg(
    smoother="identity",
    csv_name="identifications.csv",
    fixed_classes=None,          # открытый набор жителей → из p_* колонок CSV
    resident_class=None,         # нет «защищаемого» класса
    downstream_classes=None,     # нет следующего шага
    fixed_short=None,            # авто-подписи (инициалы + цифры)
    fixed_colors=None,           # палитра по индексу
    viz_topk=3,                  # top-3 персоны под тайлом (иначе N чисел не влезают)
)


def smooth_date(date_dir, **kw):
    """Обёртка с привязкой identity-конфига (для тестов и внешних вызовов)."""
    return _core_smooth_date(date_dir, IDENTITY, version=_VER, **kw)


if __name__ == "__main__":
    raise SystemExit(run_service(IDENTITY, version=_VER,
                                 desc="Темпоральное сглаживание идентификации жителей (4b)"))
