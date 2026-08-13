"""Тесты pipeline_status.py: склонение и формат ячейки «N прогонов (X кадров/файлов)»."""
from __future__ import annotations

import re
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
         "min": None, "avg": None, "max": None, "last": None,
         "cmin": None, "cavg": None, "cmax": None}
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


def test_fmt_stats_compute():
    # compute-сервисы: чистое время вычисления кадра (счёт/кадр min/avg/max мс)
    cell = ps.fmt_stats(_st(5, 10, frames=100, cmin=8, cavg=26, cmax=54),
                        has_timing=True, kind="batch")
    assert cell == "5 прогонов (10м) (100 кадров)\nмсек/кадр 8/26/54"


def test_fmt_stats_cpu_label():
    cell = ps.fmt_stats(_st(5, 10, frames=100, cmin=8, cavg=26, cmax=54),
                        has_timing=True, cpu_line="9%/29%/74%/65%", kind="batch")
    assert cell.endswith("мсек/кадр 8/26/54\n9%/29%/74%/65% (цпу)")


def test_fmt_stats_burst_receive():
    # transfer: чистое время приёма файла (I/O)
    cell = ps.fmt_stats(_st(328, 10, bursts=12, min=0.1, avg=0.1, max=0.2),
                        has_timing=True, kind="burst")
    assert cell.endswith("сек/кадр 0.1/0.1/0.2")


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
    assert "2026-07-18: 1" in out                          # дата в формате ГГГГ-ММ-ДД
    assert "2026-07-19: 1" in out
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


def test_dir_state_by_date_smoothed_counts_csv_not_viz(tmp_path):
    # сайдкар сглаживания: кропы = строки classifications_smoothed.csv, а НЕ viz-конкаты (*.jpg)
    root = tmp_path / "v4" / "inference" / "smoothed"
    d = root / "20260720"
    (d / "smooth_viz").mkdir(parents=True)
    for i in range(3):                                    # 3 viz-конката — НЕ должны считаться кропами
        (d / "smooth_viz" / f"v{i}.jpg").write_bytes(b"x")
    # 5 кропов = 5 строк данных + заголовок
    (d / "classifications_smoothed.csv").write_text(
        "crop,cam,model_class,smoothed_class\n" + "\n".join(f"c{i}.jpg,cam,1,1" for i in range(5)) + "\n",
        encoding="utf-8")
    assert ps._is_inference_images(root) is True
    out = ps.dir_state_by_date(root)
    assert "2026-07-20: 5" in out                          # 5 кропов из CSV, не 3 viz-картинки
    assert "ИТОГО: 5 (1 дат)" in out


def test_dir_state_by_date_images_still_counts_jpg(tmp_path):
    # images/<date>/ (без classifications_smoothed.csv) — считаем физические кропы *.jpg как раньше
    root = tmp_path / "v4" / "inference" / "images"
    d = root / "20260720"
    (d / "single" / "1_resident").mkdir(parents=True)
    for i in range(4):
        (d / "single" / "1_resident" / f"c{i}.jpg").write_bytes(b"x")
    out = ps.dir_state_by_date(root)
    assert "2026-07-20: 4" in out


def test_dir_state_by_date_dispatch(tmp_path):
    # не-инференс каталог → плоский счётчик (со словом «кат.»), инференс → по датам
    flat = tmp_path / "2_yolo_boxes_files" / "images"
    (flat / "run_x").mkdir(parents=True)
    (flat / "run_x" / "a.jpg").write_bytes(b"x")
    assert ps._is_inference_images(flat) is False
    assert "кат." in ps.state_for(flat)


def test_fmt_stats_zero():
    assert ps.fmt_stats(_st(0, 10), False, kind="batch") == "—"


def test_render_md_table_columns_consistent():
    # Регрессия: сырой '|' внутри ячейки (напр. в шапке '<мсек|сек>/кадр') markdown трактует
    # как разделитель колонки → таблица не рендерится. Число '|' во всех строках таблицы
    # должно совпадать (в шапке пайп обязан быть HTML-сущностью &#124;, а не литералом).
    row = {"label": "video-yolo", "state": "active",
           "input": "in", "output": "out", "log_work": "work", "log_wait": "wait",
           "stats": "1 прогон (10м) (2 кадра)\nмсек/кадр 8/16/54\n11%/11%/11%/11% (цпу)"}
    md = ps.render_md([row], "2026-08-02 00:00:00")
    table_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    counts = {ln.count("|") for ln in table_lines}
    assert len(counts) == 1, f"строки таблицы с разным числом '|' (сырой пайп в ячейке?): {counts}"


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
    assert st["cavg"] is None         # нет 'счёт/кадр' в этих строках


def test_parse_stat_lines_compute():
    # backward-compat: прежняя метка 'счёт/кадр: min/avg/max мс' (логи не перевыпущенных машин)
    lines = [
        "Jul 19 08:39:23 h bash[1]: 08:39:23  INFO  1 батч (10 кадров)  Готово. Время: 3.0 с.  счёт/кадр: 8/20/50 мс  images=/x",
        "Jul 19 08:45:10 h bash[1]: 08:45:10  INFO  1 батч (30 кадров)  Готово. Время: 9.0 с.  счёт/кадр: 5/30/90 мс  images=/x",
    ]
    st = ps.parse_stat_lines(lines, r"Готово\. Время:", has_timing=True)
    assert st["frames"] == 40
    assert st["cmin"] == 5            # min из минимумов (8, 5)
    assert st["cmax"] == 90           # max из максимумов (50, 90)
    assert st["cavg"] == 28           # взвеш. по кадрам: (20*10 + 30*30)/40 = 27.5 → 28


def test_parse_stat_lines_compute_new_label():
    # новая метка 'мсек/кадр: min/avg/max' (без суффикса ' мс' — единица в самой метке)
    lines = [
        "Jul 19 08:39:23 h bash[1]: 08:39:23  INFO  1 батч (10 кадров)  Готово. Время: 3.0 с.  мсек/кадр: 8/20/50  images=/x",
        "Jul 19 08:45:10 h bash[1]: 08:45:10  INFO  1 батч (30 кадров)  Готово. Время: 9.0 с.  мсек/кадр: 5/30/90  images=/x",
    ]
    st = ps.parse_stat_lines(lines, r"Готово\. Время:", has_timing=True)
    assert st["frames"] == 40
    assert st["cmin"] == 5
    assert st["cmax"] == 90
    assert st["cavg"] == 28


def test_compute_per_frame_log_label_and_roundtrip():
    # эмиттер пишет единую метку 'мсек/кадр' (парно к 'сек/кадр'), без старого 'счёт'
    from common.utils.camera_run import compute_per_frame_log
    s = compute_per_frame_log([8.0, 20.0, 50.0])
    assert "мсек/кадр:" in s
    assert "счёт" not in s
    # round-trip: строку эмиттера парсер понимает
    line = f"08:39:23  INFO  1 батч (3 кадров)  Готово. Время: 3.0 с.{s}"
    st = ps.parse_stat_lines([line], r"Готово\. Время:", has_timing=True)
    assert st["cmin"] == 8
    assert st["cmax"] == 50
    assert compute_per_frame_log([]) == ""


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


# ─── проводка SERVICES (сайдкар smoothed/ + шаг 4b) ───────────────────────────

def _svc(name):
    return next(s for s in ps.SERVICES if s["name"] == name)


def test_services_include_smooth_identity_step():
    # после рефакторинга на сайдкар в таблице 6 шагов, включая video-smooth-identity (4b)
    names = [s["name"] for s in ps.SERVICES]
    assert names == [
        "video-transfer", "video-yolo", "video-classify",
        "video-smooth", "video-identify", "video-smooth-identity",
    ]


def test_smooth_services_point_to_sidecar_not_images():
    # video-smooth пишет сайдкар smoothed/ (images/ НЕ мутирует), а не ин-плейс single/
    assert _svc("video-smooth")["out_dir"].name == "smoothed"
    # identify читает вход из сайдкара smoothed/, а не из физических single/<class>
    assert _svc("video-identify")["in_dir"].name == "smoothed"
    # 4b: вход — images/ Модели 2, выход — её сайдкар smoothed/; тайминга нет
    si = _svc("video-smooth-identity")
    assert si["in_dir"].name == "images" and "residents" in si["in_dir"].as_posix()
    assert si["out_dir"].name == "smoothed" and "residents" in si["out_dir"].as_posix()
    assert si["has_timing"] is False


def test_smooth_services_display_label():
    # systemd-юнит остаётся video-smooth*, но в отчёте показываем classify-/identify-smooth
    assert _svc("video-smooth")["label"] == "classify-smooth"
    assert _svc("video-smooth-identity")["label"] == "identify-smooth"
    # у не-переименованных label не задан → рендер падает на name
    assert "label" not in _svc("video-yolo")


def test_is_inference_images_covers_smoothed():
    # сайдкар smoothed/ тоже разложен по датам → разбивка по датам, а не плоский счётчик
    from pathlib import Path
    assert ps._is_inference_images(Path("/x/inference/smoothed")) is True
    assert ps._is_inference_images(Path("/x/inference/images")) is True
    assert ps._is_inference_images(Path("/x/foo/smoothed")) is False


def test_fmt_date_and_thousands():
    assert ps._fmt_date("20260720") == "2026-07-20"
    assert ps._fmt_date("dataset") == "dataset"        # не-дата не трогаем
    # тысячи отделены пробельным символом; \s нормализует и обычный, и неразрывный
    norm = lambda s: re.sub(r"\s", "_", s)
    assert norm(ps._thou(1126)) == "1_126"
    assert norm(ps._thou(4315)) == "4_315"
    assert ps._thou(999) == "999"
    assert norm(ps._thou(15967)) == "15_967"
