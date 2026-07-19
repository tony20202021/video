"""Единый список классов Модели 1 (классификатор групп людей)."""

GROUP_CLASSES: list[str] = [
    "1_resident",
    "2_delivery",
    "3_utilities",
    "4_guest",
]

RESIDENT_CLASS = "1_resident"

# Папки датасета, не являющиеся классами модели
EXTRA_DATASET_DIRS: list[str] = ["skip", "unknown", "new"]

GROUP_CLASS_COLORS: dict[str, str] = {
    "1_resident":  "#228833",
    "2_delivery":  "#0055cc",
    "3_utilities": "#770077",
    "4_guest":     "#cc7700",
    "multi":       "#aa3377",
    "uncertain":   "#dddddd",
    "unknown":     "#aaaaaa",
}

GROUP_CLASS_RU: dict[str, str] = {
    "1_resident":  "Житель",
    "2_delivery":  "Доставка",
    "3_utilities": "ЖКХ",
    "4_guest":     "Гость",
}

# Устаревшие имена → новые (None → unknown)
LEGACY_CLASS_MIGRATIONS: dict[str, str | None] = {
    "resident":  "1_resident",
    "delivery":  "2_delivery",
    "utilities": "3_utilities",
    "other":     None,
    "courier":   None,
}
