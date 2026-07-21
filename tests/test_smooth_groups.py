"""Тест пайплайн-скрипта 3b_smooth_groups: сглаживание перекладывает кропы и правит labels.json."""
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


def _make_dir(tmp_path):
    dd = tmp_path / "20260720"
    # cam_d: поток резидентов + 1 ошибочная доставка (085002) + 1 uncertain (085003)
    crops = [
        ("cam_01_9_d_20260720_085000_100000_msk.jpg", "1_resident", [0.9, 0.05, 0.02, 0.03]),
        ("cam_01_9_d_20260720_085001_100000_msk.jpg", "1_resident", [0.88, 0.05, 0.02, 0.05]),
        ("cam_01_9_d_20260720_085002_100000_msk.jpg", "2_delivery", [0.35, 0.55, 0.05, 0.05]),
        ("cam_01_9_d_20260720_085003_100000_msk.jpg", "uncertain",  [0.45, 0.30, 0.10, 0.15]),
        ("cam_01_9_d_20260720_085004_100000_msk.jpg", "1_resident", [0.9, 0.05, 0.02, 0.03]),
    ]
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
    return dd


def test_smooth_date_fixes_and_rescues(tmp_path):
    m = _load()
    dd = _make_dir(tmp_path)
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)
    assert st["eligible"] == 5
    assert st["changed"] == 2 and st["rescued_uncertain"] == 1

    # ошибочная доставка → резидент (перемещена)
    assert (dd / "single/1_resident/cam_01_9_d_20260720_085002_100000_msk.jpg").is_file()
    assert not (dd / "single/2_delivery/cam_01_9_d_20260720_085002_100000_msk.jpg").exists()
    # uncertain спасён → резидент
    assert (dd / "single/1_resident/cam_01_9_d_20260720_085003_100000_msk.jpg").is_file()
    assert not (dd / "uncertain/cam_01_9_d_20260720_085003_100000_msk.jpg").exists()

    # labels.json: все резиденты, относительные ключи, без дублей
    L = ml.load_labels(dd / "labels.json")
    assert all(v == ["1_resident"] for v in L.values())
    assert all(not k.startswith("/") for k in L)
    assert len(L) == len(set(Path(k).name for k in L))

    # аудит-CSV
    rows = list(csv.DictReader(open(dd / "classifications_smoothed.csv", encoding="utf-8")))
    assert sum(1 for r in rows if r["changed"] == "True") == 2


def test_smooth_date_gate_preserves_confident(tmp_path):
    m = _load()
    dd = _make_dir(tmp_path)
    # с гейтингом уверенных не трогаем; ошибочная доставка тут conf 0.55 < 0.8 → всё равно сгладится,
    # но проверим, что gate-параметр проходит и uncertain (низкая conf) всё ещё спасается
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=0.8, include_uncertain=True)
    assert st["rescued_uncertain"] == 1


def test_smooth_date_writes_watermark(tmp_path):
    import json
    m = _load()
    dd = _make_dir(tmp_path)
    m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)
    wm = dd / "meta" / "smooth_state.json"
    assert wm.is_file()
    st = json.loads(wm.read_text(encoding="utf-8"))
    # 5 строк CSV, все 5 eligible (3 резидента + доставка + uncertain) — identify ждёт этого watermark
    assert st["csv_rows"] == 5 and st["eligible"] == 5 and st["smoother"] == "3b"


def test_smooth_date_viz_writes_concats(tmp_path):
    m = _load()
    dd = _make_dir(tmp_path)
    # viz=True → на каждый исправленный кроп конкат-картинка в meta/smooth_viz/
    st = m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True, viz=True)
    assert st["changed"] == 2 and st.get("viz") == 2
    vdir = dd / "meta" / "smooth_viz"
    assert vdir.is_dir()
    imgs = list(vdir.glob("*.jpg"))
    assert len(imgs) == 2 and all(p.stat().st_size > 0 for p in imgs)
    # meta/ не попадает в раскладку кропов (identify/индексатор его игнорируют)
    assert not (dd / "single" / "meta").exists()
