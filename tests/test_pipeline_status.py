"""Тесты pipeline_status.py: склонение и формат ячейки «N прогонов (X кадров/файлов)»."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
import pipeline_status as ps  # noqa: E402


def test_plural_runs():
    p = lambda n: ps._plural(n, "прогон", "прогона", "прогонов")
    assert p(1) == "прогон"
    assert p(2) == "прогона"
    assert p(4) == "прогона"
    assert p(5) == "прогонов"
    assert p(11) == "прогонов"   # 11-14 — исключение
    assert p(21) == "прогон"
    assert p(22) == "прогона"


def test_plural_frames():
    p = lambda n: ps._plural(n, "кадр", "кадра", "кадров")
    assert p(0) == "кадров"
    assert p(1) == "кадр"
    assert p(262) == "кадра"     # …2 → few
    assert p(330) == "кадров"    # …0 → many


def _st(count, window, frames=0, bursts=0, **kw):
    d = {"count": count, "window": window, "frames": frames, "bursts": bursts,
         "min": None, "avg": None, "max": None, "last": None}
    d.update(kw)
    return d


def test_fmt_stats_batch():
    # yolo/classify/identify: N прогонов (X кадров) — X из лога «1 батч (X кадров)»
    assert ps.fmt_stats(_st(9, 60, frames=330), False, kind="batch") \
        == "9 прогонов (1ч) (330 кадров)"
    assert ps.fmt_stats(_st(1, 10, frames=1), False, kind="batch") \
        == "1 прогон (10м) (1 кадр)"
    # даже если ничего не найдено — прогоны считаются, кадры показываются
    assert ps.fmt_stats(_st(10, 10, frames=0), False, kind="batch") \
        == "10 прогонов (10м) (0 кадров)"


def test_fmt_stats_burst():
    # transfer: N всплесков приёма (прогонов) (X принятых файлов)
    assert ps.fmt_stats(_st(328, 10, bursts=12), False, kind="burst") \
        == "12 прогонов (10м) (328 файлов)"


def test_fmt_stats_timing_unit():
    # единица «на 1кадр» (тайминг на кадр) — в конце строки; ЦПУ помечен «(цпу)»
    cell = ps.fmt_stats(_st(5, 10, frames=100, min=0.2, avg=0.4, max=1.7, last=0.3),
                        has_timing=True, kind="batch")
    assert cell == "5 прогонов (10м) (100 кадров)\n0.2с/0.4с/1.7с/0.3с (на 1кадр)"


def test_fmt_stats_cpu_label():
    cell = ps.fmt_stats(_st(5, 10, frames=100, min=0.2, avg=0.4, max=1.7, last=0.3),
                        has_timing=True, cpu_line="9%/29%/74%/65%", kind="batch")
    assert cell.endswith("0.2с/0.4с/1.7с/0.3с (на 1кадр)\n9%/29%/74%/65% (цпу)")


def test_fmt_stats_burst_unit_file():
    # transfer: тайминг на 1 файл (строка = 1 файл)
    cell = ps.fmt_stats(_st(328, 10, bursts=12, min=0.1, avg=0.1, max=0.2, last=0.1),
                        has_timing=True, kind="burst")
    assert cell.endswith("0.1с/0.1с/0.2с/0.1с (на 1файл)")


# ─── dir_state_by_date (разбивка инференса по датам) ──────────────────────────

def test_dir_state_by_date(tmp_path):
    root = tmp_path / "v1" / "inference" / "images"
    for d in ("20260718", "20260719"):
        (root / d / "single" / "1_resident").mkdir(parents=True)
        (root / d / "single" / "1_resident" / "a.jpg").write_bytes(b"x")
    (root / "dataset" / "p1").mkdir(parents=True)          # не-дата → показать как лишнее
    (root / "dataset" / "p1" / "z.jpg").write_bytes(b"x")
    assert ps._is_inference_images(root) is True
    out = ps.dir_state_by_date(root)
    assert "20260718: 1" in out
    assert "20260719: 1" in out
    assert "ИТОГО: 2 (2 дат)" in out                       # итог — только по датам
    assert "[!] dataset: 1 (не дата)" in out               # dataset/ виден отдельной строкой


def test_dir_state_by_date_empty_nondate_hidden(tmp_path):
    # пустая не-дата (без картинок) не засоряет — показываем только с данными
    root = tmp_path / "v1" / "inference" / "images"
    (root / "20260719").mkdir(parents=True)
    (root / "20260719" / "a.jpg").write_bytes(b"x")
    (root / "tmp_empty").mkdir()
    out = ps.dir_state_by_date(root)
    assert "tmp_empty" not in out
    assert "ИТОГО: 1 (1 дат)" in out


def test_dir_state_by_date_dispatch(tmp_path):
    # не-инференс каталог → плоский счётчик (со словом «кат.»), инференс → по датам
    flat = tmp_path / "2_yolo_boxes_files" / "images"
    (flat / "run_x").mkdir(parents=True)
    (flat / "run_x" / "a.jpg").write_bytes(b"x")
    assert ps._is_inference_images(flat) is False
    assert "кат." in ps.state_for(flat)


def test_fmt_stats_zero():
    assert ps.fmt_stats(_st(0, 10), False, kind="batch") == "—"


# ─── parse_stat_lines (разбор строк журнала) ──────────────────────────────────

def test_parse_stat_lines_batch():
    # yolo/classify/identify: '1 батч (X кадров)  Готово. Время: N с.'
    lines = [
        "Jul 19 08:39:23 h bash[1]: 08:39:23  INFO      1 батч (23 кадров)  Готово. Время: 5.8 с.  images=/x",
        "Jul 19 08:45:10 h bash[1]: 08:45:10  INFO      1 батч (7 кадров)  Готово. Время: 12.0 с.  images=/x",
        "Jul 19 08:45:05 h bash[1]: 08:45:05  INFO        (нет детекций YOLO)",   # не совпадает
    ]
    st = ps.parse_stat_lines(lines, r"Готово\. Время:", has_timing=True)
    assert st["count"] == 2           # 2 прогона (Готово)
    assert st["frames"] == 30         # 23 + 7 кадров (суммируется)
    # тайминг НА 1 КАДР: 5.8/23=0.25, 12.0/7=1.71 (округл. 2 знака)
    assert st["min"] == 0.25
    assert st["max"] == 1.71
    assert st["avg"] == 0.98          # среднее (0.2522+1.7143)/2
    assert st["last"] == 1.71         # последняя строка: 12.0/7


def test_parse_stat_lines_bursts():
    # transfer: [recv] с таймстампами; пауза > BURST_GAP_SEC делит на всплески
    lines = [
        "Jul 19 08:39:01 h bash[1]: 08:39:01  INFO      [recv] diff/x.jpg  Готово. Время: 0.1 с.",
        "Jul 19 08:39:02 h bash[1]: 08:39:02  INFO      [recv] diff/y.jpg  Готово. Время: 0.1 с.",
        "Jul 19 08:39:03 h bash[1]: 08:39:03  INFO      [recv] diff/z.jpg  Готово. Время: 0.1 с.",
        "Jul 19 08:45:00 h bash[1]: 08:45:00  INFO      [recv] diff/w.jpg  Готово. Время: 0.1 с.",
    ]
    st = ps.parse_stat_lines(lines, r"\[recv\].*Готово\. Время:", has_timing=True)
    assert st["count"] == 4           # 4 принятых файла
    assert st["bursts"] == 2          # 3 подряд + 1 после паузы > 15с
    assert st["frames"] == 0          # у transfer нет 'кадров'


def test_parse_stat_lines_empty():
    st = ps.parse_stat_lines(["нет совпадений тут"], r"Готово\. Время:", has_timing=True)
    assert st["count"] == 0 and st["frames"] == 0 and st["bursts"] == 0
    assert st["avg"] is None
