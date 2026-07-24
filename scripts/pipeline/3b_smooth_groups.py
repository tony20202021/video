"""
3b — темпоральное сглаживание классов Модели 1 (группы). Тонкая обёртка над common.utils.smooth_core.

images/ НЕ мутируется. Выход — сайдкар <inference>/smoothed/<date>/ (classifications_smoothed.csv,
smooth_state.json, smooth_viz/). Идентичное ядро использует 4b_smooth_identity.py (жители).

Usage:
    python scripts/pipeline/3b_smooth_groups.py .data/groups/v4/inference/images --once
    python scripts/pipeline/3b_smooth_groups.py <root> --prob-window 0 --gate 0 --p-stay 0.9
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.classes import GROUP_CLASSES, RESIDENT_CLASS, GROUP_CLASS_COLORS
from common.utils.smooth_core import SmoothCfg, run_service, smooth_date as _core_smooth_date

GUEST_CLASS = "4_guest"
_SHORT = {"1_resident": "res", "2_delivery": "del", "3_utilities": "utl", "4_guest": "gst"}

try:
    _VER = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    _VER = ""

GROUPS = SmoothCfg(
    smoother="groups",
    csv_name="classifications.csv",
    fixed_classes=GROUP_CLASSES,
    resident_class=RESIDENT_CLASS,
    downstream_classes={RESIDENT_CLASS, GUEST_CLASS},   # кропы на identify (Модель 2)
    fixed_short=_SHORT,
    fixed_colors=GROUP_CLASS_COLORS,
    viz_topk=len(GROUP_CLASSES),
)


def smooth_date(date_dir, **kw):
    """Обёртка с привязкой groups-конфига (для тестов и внешних вызовов)."""
    return _core_smooth_date(date_dir, GROUPS, version=_VER, **kw)


if __name__ == "__main__":
    raise SystemExit(run_service(GROUPS, version=_VER,
                                 desc="Темпоральное сглаживание классов Модели 1 (3b, группы)"))
