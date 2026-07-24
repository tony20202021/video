"""Тест 3b_smooth_groups: сглаживание НЕ мутирует images/, пишет сайдкар smoothed/<date>/."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.utils import multilabel as ml
from common.utils.classes import GROUP_CLASSES as C


def _load():
    p = REPO_ROOT / "scripts" / "pipeline" / "3b_smooth_groups.py"
    spec = importlib.util.spec_from_file_location("smooth3b", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_dir(dd: Path, crops):
    for name, oc, _ in crops:
        d = dd / "single" / oc if oc in C else dd / oc
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"x")
    ml.save_labels(dd / "labels.json",
                   {f"single/{oc}/{n}": [oc] for n, oc, _ in crops if oc in C})
    fields = (["mono_s", "ts_epoch", "run_name", "sub_run", "cam", "crop", "group",
               "group_conf", "conf_2nd", "margin"] + [f"p_{c}" for c in C]
              + ["classes", "n_classes", "to_identify", "out_class"])
    with open(dd / "classifications.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, oc, pr in crops:
            row = {k: "" for k in fields}
            row.update(crop=name, out_class=oc,
                       group=(oc if oc in C else C[pr.index(max(pr))]))
            for c, p in zip(C, pr):
                row[f"p_{c}"] = p
            w.writerow(row)


def _make_dir(tmp_path):
    # раскладка как в реальном инференсе: <inference>/images/<date>/  → сайдкар <inference>/smoothed/<date>/
    dd = tmp_path / "images" / "20260720"
    _write_dir(dd, [
        ("cam_01_9_d_20260720_085000_100000_msk.jpg", "1_resident", [0.9, 0.05, 0.02, 0.03]),
        ("cam_01_9_d_20260720_085001_100000_msk.jpg", "1_resident", [0.88, 0.05, 0.02, 0.05]),
        ("cam_01_9_d_20260720_085002_100000_msk.jpg", "2_delivery", [0.35, 0.55, 0.05, 0.05]),
        ("cam_01_9_d_20260720_085003_100000_msk.jpg", "uncertain",  [0.45, 0.30, 0.10, 0.15]),
        ("cam_01_9_d_20260720_085004_100000_msk.jpg", "1_resident", [0.9, 0.05, 0.02, 0.03]),
    ])
    return dd


def _smoothed(tmp_path):
    return tmp_path / "smoothed" / "20260720"


def _sm_map(sm_dir):
    return {r["crop"]: r for r in
            csv.DictReader(open(sm_dir / "classifications_smoothed.csv", encoding="utf-8"))}


def test_smooth_writes_sidecar_not_mutates_images(tmp_path):
    m = _load()
    dd = _make_dir(tmp_path)
    sm = _smoothed(tmp_path)
    # снимок images/ до сглаживания
    before = sorted(p.relative_to(dd).as_posix() for p in dd.rglob("*") if p.is_file())
    labels_before = (dd / "labels.json").read_text(encoding="utf-8")

    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)
    assert st["eligible"] == 5
    assert st["changed"] == 2 and st["rescued_uncertain"] == 1 and st["moved"] == 0

    # images/ НЕ изменился: те же файлы, тот же labels.json, никаких сглаж-артефактов
    after = sorted(p.relative_to(dd).as_posix() for p in dd.rglob("*") if p.is_file())
    assert after == before
    assert (dd / "labels.json").read_text(encoding="utf-8") == labels_before
    assert not (dd / "classifications_smoothed.csv").exists()
    assert not (dd / "meta").exists()
    # ошибочная доставка и uncertain — на СВОИХ местах (не переложены)
    assert (dd / "single/2_delivery/cam_01_9_d_20260720_085002_100000_msk.jpg").is_file()
    assert (dd / "uncertain/cam_01_9_d_20260720_085003_100000_msk.jpg").is_file()

    # сглаженный класс — в сайдкаре smoothed/<date>/
    byname = _sm_map(sm)
    assert byname["cam_01_9_d_20260720_085002_100000_msk.jpg"]["smoothed_class"] == "1_resident"
    assert byname["cam_01_9_d_20260720_085003_100000_msk.jpg"]["smoothed_class"] == "1_resident"
    assert sum(1 for r in byname.values() if r["changed"] == "True") == 2


def test_smooth_date_smooths_multiperson(tmp_path):
    # .plan: M>1 кадр (p2of2) больше НЕ исключается — участвует в окне и сам исправляется
    m = _load()
    dd = tmp_path / "images" / "20260720"
    _write_dir(dd, [
        ("cam_01_9_d_20260720_085000_100000_msk_p1of1.jpg", "1_resident", [0.9, 0.05, 0.02, 0.03]),
        ("cam_01_9_d_20260720_085001_100000_msk_p2of2.jpg", "2_delivery", [0.35, 0.55, 0.05, 0.05]),
        ("cam_01_9_d_20260720_085002_100000_msk_p1of1.jpg", "1_resident", [0.9, 0.05, 0.02, 0.03]),
    ])
    st = m.smooth_date(dd, gap=5, p_stay=0.95, gate=None, include_uncertain=True, prob_window_sec=3.0)
    assert st["eligible"] == 3          # все 3, включая M>1 (p2of2)
    # images/ не тронут: p2of2 остаётся в single/2_delivery/
    assert (dd / "single/2_delivery/cam_01_9_d_20260720_085001_100000_msk_p2of2.jpg").is_file()
    # но сглаженный класс = resident в сайдкаре
    byname = _sm_map(_smoothed(tmp_path))
    assert byname["cam_01_9_d_20260720_085001_100000_msk_p2of2.jpg"]["smoothed_class"] == "1_resident"


def test_smooth_date_errors_vs_truth(tmp_path):
    # ground-truth → считаем ошибки (сглаж∉истины), в т.ч. НЕ исправленные сглаживанием
    m = _load()
    dd = _make_dir(tmp_path)   # доставка@085002 сглаживается в resident; всё прочее resident
    truth = {
        "cam_01_9_d_20260720_085002_100000_msk.jpg": ["2_delivery"],  # сглажено в resident → ОШИБКА (изменённая)
        "cam_01_9_d_20260720_085000_100000_msk.jpg": ["4_guest"],     # остался resident → ОШИБКА неисправленная
    }
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True, truth=truth)
    assert st["errors"] == 2
    assert st["errors_unfixed"] == 1     # 085000 сглаживание не трогало


def test_smooth_date_skips_unchanged(tmp_path):
    # второй поллинг без новых кропов и с тем же конфигом → дату НЕ пересглаживаем
    m = _load()
    dd = _make_dir(tmp_path)
    sm = _smoothed(tmp_path)
    st1 = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)
    assert not st1.get("skipped")
    mtime = (sm / "classifications_smoothed.csv").stat().st_mtime_ns
    # тот же конфиг/версия, CSV не рос → ПРОПУСК (CSV не переписан)
    st2 = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)
    assert st2.get("skipped") is True
    assert (sm / "classifications_smoothed.csv").stat().st_mtime_ns == mtime
    # смена конфига (gap) → снова пересглаживание (не пропуск)
    st3 = m.smooth_date(dd, gap=5, p_stay=0.95, gate=None, include_uncertain=True)
    assert not st3.get("skipped")


def test_smooth_date_gate_preserves_confident(tmp_path):
    m = _load()
    dd = _make_dir(tmp_path)
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=0.8, include_uncertain=True)
    assert st["rescued_uncertain"] == 1


def test_smooth_date_writes_watermark(tmp_path):
    import json
    m = _load()
    dd = _make_dir(tmp_path)
    m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)
    wm = _smoothed(tmp_path) / "smooth_state.json"       # watermark в сайдкаре, НЕ в images/
    assert wm.is_file()
    assert not (dd / "meta" / "smooth_state.json").exists()
    st = json.loads(wm.read_text(encoding="utf-8"))
    assert st["csv_rows"] == 5 and st["eligible"] == 5 and st["smoother"] == "groups"


def test_smooth_date_viz_writes_concats(tmp_path):
    m = _load()
    dd = _make_dir(tmp_path)
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True, viz=True)
    assert st["changed"] == 2 and st.get("viz") == 1
    vdir = _smoothed(tmp_path) / "smooth_viz"            # конкаты в сайдкаре, НЕ в images/
    assert vdir.is_dir()
    imgs = list(vdir.glob("*.jpg"))
    assert len(imgs) == 1 and all(p.stat().st_size > 0 for p in imgs)
    assert not (dd / "meta").exists()                    # images/ без meta/
