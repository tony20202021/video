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


def test_smooth_writes_csv_smoothed_class_not_labels(tmp_path):
    # сглаживатель пишет smoothed_class в CSV (col4) + rel/p_*, но labels.json НЕ пишет
    # (labels.json — файл РУЧНОЙ разметки: 2_label_ui сидит стартовые метки из CSV, когда его нет)
    import csv as _csv
    m = _load()
    dd = _make_dir(tmp_path)
    sm = _smoothed(tmp_path)
    m.smooth_date(dd, gap=60, p_stay=0.95, gate=None, include_uncertain=True)

    # labels.json сглаживатель НЕ создаёт
    assert not (sm / "labels.json").exists()

    # smoothed_class в col4 (identify.sh читает $4); CSV обогащён rel + p_<class>
    rel_delivery = "single/2_delivery/cam_01_9_d_20260720_085002_100000_msk.jpg"
    header = next(_csv.reader(open(sm / "classifications_smoothed.csv", encoding="utf-8")))
    assert header[3] == "smoothed_class"
    assert "rel" in header and "p_1_resident" in header and "p_4_guest" in header
    row = _sm_map(sm)["cam_01_9_d_20260720_085002_100000_msk.jpg"]
    assert row["smoothed_class"] == "1_resident"      # доставка сглажена в resident
    assert row["rel"] == rel_delivery
    assert float(row["p_2_delivery"]) > 0             # probs модели сохранены


def test_hard_vote_one_class_per_visit(tmp_path):
    # SMOOTH_HARD_VOTE: весь визит → 1 класс = argmax среднего probs (доставка-одиночка тонет в резиденте)
    m = _load()
    dd = _make_dir(tmp_path)   # 4×resident + 1×delivery@085002 в одном визите (gap большой)
    sm = _smoothed(tmp_path)
    st = m.smooth_date(dd, gap=60, p_stay=0.9, gate=None, include_uncertain=True, hard_vote=True)
    assert st["eligible"] == 5
    byname = _sm_map(sm)
    # среднее probs визита → resident доминирует → ВСЕ 5 кадров = resident (в т.ч. доставка и uncertain)
    assert {r["smoothed_class"] for r in byname.values()} == {"1_resident"}
    # cfg_sig различает hard_vote → повторный прогон без флага пересглаживает (не пропуск)
    st2 = m.smooth_date(dd, gap=60, p_stay=0.9, gate=None, include_uncertain=True, hard_vote=False)
    assert not st2.get("skipped")


def test_hard_vote_merge_zones_separate_cameras(tmp_path):
    # hard_vote + merge_zones: зоны d/u ОДНОЙ камеры голосуют одним потоком; РАЗНЫЕ физкамеры — раздельно.
    m = _load()
    dd = tmp_path / "images" / "20260720"
    _write_dir(dd, [
        # cam_01: d-зона резидент + u-зона доставка → один поток, среднее → резидент доминирует
        ("cam_01_9_d_20260720_090000_100000_msk.jpg", "1_resident", [0.9, 0.05, 0.03, 0.02]),
        ("cam_01_9_u_20260720_090001_100000_msk.jpg", "2_delivery", [0.3, 0.6, 0.05, 0.05]),
        # cam_02: отдельная физкамера, сплошь доставка → её голос НЕ смешивается с cam_01
        ("cam_02_9_d_20260720_090000_100000_msk.jpg", "2_delivery", [0.1, 0.85, 0.03, 0.02]),
        ("cam_02_9_d_20260720_090001_100000_msk.jpg", "2_delivery", [0.12, 0.83, 0.03, 0.02]),
    ])
    m.smooth_date(dd, gap=60, p_stay=0.9, gate=None, include_uncertain=True,
                  hard_vote=True, merge_zones=True)
    byname = _sm_map(_smoothed(tmp_path))
    # cam_01 (d+u склеены): среднее resident>delivery → оба кадра резидент
    assert byname["cam_01_9_d_20260720_090000_100000_msk.jpg"]["smoothed_class"] == "1_resident"
    assert byname["cam_01_9_u_20260720_090001_100000_msk.jpg"]["smoothed_class"] == "1_resident"
    # cam_02 голосует отдельно → доставка сохранена (склейка НЕ утопила её в резиденте cam_01)
    assert byname["cam_02_9_d_20260720_090000_100000_msk.jpg"]["smoothed_class"] == "2_delivery"


def test_merge_zones_cam_key():
    # SMOOTH_MERGE_ZONES: зоны d/u ОДНОЙ камеры → один ключ камеры; разные физ. камеры — раздельно
    from common.utils.smooth_core import _parse_cam_t
    n_d = "cam_01_9_d_20260720_085017_100000_msk.jpg"
    n_u = "cam_01_9_u_20260720_085017_100000_msk.jpg"
    assert _parse_cam_t(n_d)[0] == "cam_01_9_d"                    # выкл (по умолч.) — зона в ключе
    assert _parse_cam_t(n_d, merge_zones=True)[0] == "cam_01_9"   # склейка снимает зону d
    assert _parse_cam_t(n_u, merge_zones=True)[0] == "cam_01_9"   # u → та же камера, что d
    assert _parse_cam_t("cam_02_9_d_20260720_085017_100000_msk.jpg",
                        merge_zones=True)[0] == "cam_02_9"         # другая физ. камера — отдельный ключ


def test_split_zone_merged_concat_name():
    # имя merged-конката: зона d/u перед датой → общий суффикс из зон прохода (cam_01_9_du)
    from common.utils.smooth_core import _split_zone
    n = "cam_01_9_d_20260815_025614_889277_msk_diff4.7_p1of1_conf0.88_b341-1-517-225"
    pre, z, suf = _split_zone(n)
    assert pre == ["cam", "01", "9"] and z == "d" and suf[0] == "20260815"
    zs = sorted({zz for nm in [n, n.replace("_d_", "_u_")] for zz in [_split_zone(nm)[1]] if zz})
    assert "_".join(pre + ["".join(zs)] + suf).startswith("cam_01_9_du_20260815_025614")
    assert _split_zone("cam_02_20260815_010101_1_msk")[1] is None   # без суффикса зоны — не ломается


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
