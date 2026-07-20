"""Обучение классификатора группы (Модель 1 — MobileNetV3-Small → ONNX).

MULTI-LABEL: кроп YOLO (bbox+паддинг) часто содержит несколько людей РАЗНЫХ классов,
поэтому цель — мультихот-вектор (0/1 на каждый класс), функция потерь BCEWithLogitsLoss,
вывод sigmoid + порог на каждый класс. Старые single-label датасеты тоже читаются
(1 класс → 1-элементный мультихот), но обученная модель уже multi-label (манифест multi_label=true).

Входные данные (любой из форматов):
  - v4 датасет: dataset/ с single/<class>/img.jpg + multi/img.jpg + labels.json ({img:[classes]})
  - labels.json из 2_label_ui (пути относительно REPO_ROOT), формат v1 {img:'class'} или v2 {img:[classes]}
  - folder-based (старое): подпапки по классам 1_resident/… (каждый файл = 1 класс)

Выходные данные:
  - .models/classify/v4_1.onnx   — ONNX (датасет v4, 1-й прогон)
  - .models/classify/v4_1.json   — манифест: multi_label=true + пороги по классам (подобраны на валидации)
  - .models/classify/backbone.pt — веса backbone для инициализации Модели 2
  - training_results.json        — метрики (per-class precision/recall/F1/AP, mAP)

Требования:
  pip install torch torchvision

Usage:
    python scripts/train/3_train_groups.py --data .data/groups/v4/dataset --epochs 30
    python scripts/train/3_train_groups.py --data /path/to/labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.classes import GROUP_CLASSES as CLASSES, EXTRA_DATASET_DIRS
from common.utils import multilabel as ml
from ml.versions import (
    classify_model_path,
    models_dir,
    next_classify_model_tag,
    resolve_dataset_version,
    write_classify_manifest,
)
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_THR_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _detect_base(lab_map: dict, candidates: list[Path]) -> Path:
    """Из каких базовых директорий пути в labels.json резолвятся в файлы (v4 dir vs REPO_ROOT)."""
    keys = list(lab_map.keys())[:20]
    best, best_hits = candidates[0], -1
    for base in candidates:
        hits = sum(1 for k in keys if (base / k).is_file())
        if hits > best_hits:
            best, best_hits = base, hits
    return best


def _load_dataset(data_path: Path) -> tuple[Path, list[dict]]:
    """Распаковывает zip если нужно, возвращает (base_dir, labels).

    labels — список {"image": rel_path, "classes": [список классов]} (мультихот-совместимый).
    Старые форматы (folder-based, single-label labels.json) → 1-элементный список классов.
    """
    if data_path.suffix == ".zip":
        extract_dir = data_path.parent / data_path.stem
        with zipfile.ZipFile(data_path) as zf:
            zf.extractall(extract_dir)
        data_path = extract_dir

    # Folder-based (старое): подпапки по классам, БЕЗ labels.json.
    # v4 (single/<class>/ + multi/) сюда НЕ попадает — верхние папки 'single'/'multi', не классы.
    if data_path.is_dir():
        class_dirs = [d for d in data_path.iterdir()
                      if d.is_dir() and d.name in CLASS_TO_IDX]
        if class_dirs and not (data_path / "labels.json").is_file():
            labels = []
            for cls_dir in sorted(class_dirs):
                for f in sorted(cls_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        labels.append({"image": str(f.relative_to(data_path)),
                                       "classes": [cls_dir.name]})
            return data_path, labels

    # labels.json (v4 или 2_label_ui; формат v1/v2 — через multilabel.load_labels)
    if data_path.is_file() and data_path.name == "labels.json":
        labels_file, base_default = data_path, data_path.parent
    else:
        labels_file, base_default = data_path / "labels.json", data_path
    if not labels_file.is_file():
        raise FileNotFoundError(
            f"labels.json не найден в {data_path}. "
            f"Передайте v4-датасет (single/+multi/+labels.json) или папку с подпапками "
            f"по классам ({', '.join(CLASSES)})"
        )

    lab_map = ml.load_labels(labels_file)          # {img: [classes]}, старый v1 → [class]
    base = _detect_base(lab_map, [base_default, REPO_ROOT])
    _extra = set(EXTRA_DATASET_DIRS)               # skip/unknown/new — исключаем из обучения
    labels = [{"image": img, "classes": [c for c in cls if c in CLASS_TO_IDX]}
              for img, cls in lab_map.items()
              if not any(c in _extra for c in cls)]
    return base, labels


def _build_model(num_classes: int, pretrained: bool = True):
    try:
        import torchvision.models as M
        import torch.nn as nn
    except ImportError:
        print("Нужен PyTorch: pip install torch torchvision", file=sys.stderr)
        return None

    model = M.mobilenet_v3_small(
        weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.classifier[3].in_features
    model.classifier[3] = __import__("torch").nn.Linear(in_features, num_classes)
    return model


def _collect_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Инференс по loader. Возвращает (y_true, y_score) — массивы N×C (sigmoid-вероятности)."""
    import torch
    ys, ss = [], []
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            out = torch.sigmoid(model(imgs.to(device)))
            ss.append(out.cpu().numpy())
            ys.append(targets.numpy())
    if not ys:
        return np.zeros((0, NUM_CLASSES)), np.zeros((0, NUM_CLASSES))
    return np.concatenate(ys), np.concatenate(ss)


def _average_precision(y_true_col: np.ndarray, y_score_col: np.ndarray) -> float:
    """Average Precision (площадь под PR-кривой) для одного класса, без sklearn."""
    pos = float(y_true_col.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-y_score_col)
    yt = y_true_col[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1.0 - yt)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / pos
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def _prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, int, int, int]:
    """precision, recall, f1, tp, fp, fn для булевых массивов pred/truth."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def _tune_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Подбор порога на КАЖДЫЙ класс по максимуму F1 на валидации. Класс без позитивов → 0.5."""
    thr: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        truth = y_true[:, i] == 1
        if truth.sum() == 0:
            thr[cls] = ml.DEFAULT_THRESHOLD
            continue
        best_t, best_f1 = ml.DEFAULT_THRESHOLD, -1.0
        for t in _THR_GRID:
            _, _, f1, *_ = _prf(y_score[:, i] >= t, truth)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[cls] = best_t
    return thr


def _multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        thresholds: dict[str, float]) -> dict:
    """Per-class precision/recall/F1/AP при заданных порогах + macro/micro F1, mAP, subset-accuracy."""
    by_class: dict[str, dict] = {}
    macro_f1 = 0.0
    map_sum = 0.0
    micro_tp = micro_fp = micro_fn = 0
    pred_all = np.zeros_like(y_true, dtype=bool)

    for i, cls in enumerate(CLASSES):
        t = float(thresholds.get(cls, ml.DEFAULT_THRESHOLD))
        truth = y_true[:, i] == 1
        pred = y_score[:, i] >= t
        pred_all[:, i] = pred
        precision, recall, f1, tp, fp, fn = _prf(pred, truth)
        ap = _average_precision(y_true[:, i], y_score[:, i])
        by_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap": round(ap, 4),
            "threshold": round(t, 2),
            "support": int(truth.sum()),
            "tp": tp, "fp": fp, "fn": fn,
        }
        macro_f1 += f1
        map_sum += ap
        micro_tp += tp; micro_fp += fp; micro_fn += fn

    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    n = len(y_true)
    subset_acc = float((pred_all == (y_true == 1)).all(axis=1).mean()) if n else 0.0

    return {
        "total_samples": n,
        "macro_f1": round(macro_f1 / NUM_CLASSES, 4),
        "micro_f1": round(micro_f1, 4),
        "mAP": round(map_sum / NUM_CLASSES, 4),
        "subset_accuracy": round(subset_acc, 4),
        "by_class": by_class,
    }


def _export_onnx(model, path: Path, device) -> None:
    """ONNX export без изменения режима модели."""
    import torch
    was_training = model.training
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["input"], output_names=["output"],
            opset_version=18, dynamo=False,
        )
    if was_training:
        model.train()


def train(
    data_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    class_weights: bool = True,
    weighted_sampling: bool = False,
    output_dir: Path,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Нужен PyTorch: pip install torch torchvision pillow", file=sys.stderr)
        return {}

    base_dir, labels = _load_dataset(data_path)
    dataset_version, _dataset_dir = resolve_dataset_version(data_path)
    try:
        dataset_path_rel = str(data_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        dataset_path_rel = str(data_path)
    model_tag = next_classify_model_tag(dataset_version)
    onnx_path = classify_model_path(model_tag)
    progress_path = onnx_path.with_name(f"{model_tag}_progress.json")
    print(f"Датасет: {dataset_version or '(без версии)'}  →  модель: {model_tag}", flush=True)

    # Валидные: файл существует (метки-классы уже отфильтрованы по CLASS_TO_IDX в _load_dataset).
    # Пустой список классов = размечено «ни одного» → all-negative пример (оставляем).
    valid = [lb for lb in labels if (base_dir / lb["image"]).is_file()]
    if not valid:
        print("Нет валидных изображений.", file=sys.stderr)
        return {}

    n_multi = sum(1 for lb in valid if len(lb["classes"]) > 1)
    n_empty = sum(1 for lb in valid if len(lb["classes"]) == 0)
    print(f"Изображений: {len(valid)}  (мульти-класс: {n_multi}, без класса: {n_empty})", flush=True)
    class_counts = {c: 0 for c in CLASSES}
    for lb in valid:
        for c in lb["classes"]:
            class_counts[c] += 1
    for cls in CLASSES:
        print(f"  {cls}: {class_counts[cls]}", flush=True)

    _train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])
    _val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class CropDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            img = Image.open(base_dir / item["image"]).convert("RGB")
            target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            for c in item["classes"]:
                target[CLASS_TO_IDX[c]] = 1.0
            return self.transform(img), target

    # Стратифицированное разбиение по КОМБИНАЦИИ классов (мульти-комбо представлены в train и val)
    import random as _random
    by_combo: dict[tuple, list] = {}
    for lb in valid:
        by_combo.setdefault(tuple(sorted(lb["classes"])), []).append(lb)
    train_items, val_items = [], []
    for combo, items in by_combo.items():
        shuffled = items[:]
        _random.shuffle(shuffled)
        n_v = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
        val_items.extend(shuffled[:n_v])
        train_items.extend(shuffled[n_v:])
    n_train, n_val = len(train_items), len(val_items)

    # Позитивы по классам В TRAIN (для pos_weight) — баланс мультихот
    train_counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for lb in train_items:
        for c in lb["classes"]:
            train_counts[CLASS_TO_IDX[c]] += 1

    train_ds = CropDataset(train_items, _train_tf)

    if weighted_sampling:
        # вес примера = макс. инв.частота среди его классов (upweight редких комбинаций)
        w_per_class = n_train / (NUM_CLASSES * np.where(train_counts > 0, train_counts, 1))
        sample_weights = []
        for lb in train_items:
            idxs = [CLASS_TO_IDX[c] for c in lb["classes"]]
            sample_weights.append(float(max((w_per_class[i] for i in idxs), default=1.0)))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_items), replacement=True)
        print("WeightedRandomSampler включён", flush=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(CropDataset(val_items, _val_tf), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}", flush=True)

    model = _build_model(NUM_CLASSES, pretrained=True)
    if model is None:
        return {}
    model = model.to(device)

    # Фаза 1: только голова
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # BCEWithLogitsLoss (мультихот) + pos_weight = neg/pos на класс (при дисбалансе)
    if class_weights:
        pos = np.where(train_counts > 0, train_counts, 1)
        pos_weight = np.clip((n_train - pos) / pos, 1.0, 20.0)
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        print(f"pos_weight: { {c: round(float(pw[i]), 2) for i, c in enumerate(CLASSES)} }", flush=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        # Фаза 2: вся сеть с середины обучения
        if epoch == epochs // 2 + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            print(f"  epoch {epoch}: разморозка всех весов", flush=True)

        model.train()
        train_loss, train_total = 0.0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
            train_total += len(imgs)

        # Валидация: macro-F1 при пороге 0.5 (для выбора лучшей модели)
        yv_true, yv_score = _collect_scores(model, val_loader, device)
        val_metrics = _multilabel_metrics(yv_true, yv_score,
                                          {c: ml.DEFAULT_THRESHOLD for c in CLASSES})
        val_f1 = val_metrics["macro_f1"]
        val_map = val_metrics["mAP"]
        train_loss_avg = train_loss / train_total if train_total else 0.0
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": round(train_loss_avg, 4),
                        "val_macro_f1": val_f1, "val_mAP": val_map})

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best.pt")
            _export_onnx(model, onnx_path, device)
            print(f"  epoch {epoch:3d}/{epochs}  loss={train_loss_avg:.4f}  val_F1={val_f1:.3f}  mAP={val_map:.3f}  ← best → {onnx_path.name}", flush=True)
        else:
            print(f"  epoch {epoch:3d}/{epochs}  loss={train_loss_avg:.4f}  val_F1={val_f1:.3f}  mAP={val_map:.3f}", flush=True)

        progress_path.write_text(json.dumps({
            "model_tag": model_tag,
            "epoch": epoch,
            "epochs": epochs,
            "best_val_macro_f1": round(best_macro_f1, 4),
            "history": history,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Лучшие веса → финальный экспорт
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    model.eval()
    _export_onnx(model, onnx_path, device)

    # Backbone для инициализации Модели 2 (5_train_residents.py)
    backbone_path = models_dir() / "backbone.pt"
    backbone_state = {k: v for k, v in model.state_dict().items()
                      if k.startswith("features.")}
    torch.save(backbone_state, backbone_path)
    print(f"Backbone (для Модели 2): {backbone_path}", flush=True)

    (output_dir / "best.pt").unlink(missing_ok=True)

    # Подбор порогов по КЛАССАМ на валидации (дёшево — val-скоры уже нужны)
    print("\nПодбор порогов по классам (валидация)…", flush=True)
    yv_true, yv_score = _collect_scores(model, val_loader, device)
    thresholds = _tune_thresholds(yv_true, yv_score)
    print(f"  пороги: { {c: thresholds[c] for c in CLASSES} }", flush=True)
    eval_val = _multilabel_metrics(yv_true, yv_score, thresholds)
    print(f"  val:   macro_F1={eval_val['macro_f1']:.3f}  mAP={eval_val['mAP']:.3f}  subset_acc={eval_val['subset_accuracy']:.3f}", flush=True)

    metrics = {
        "model_tag": model_tag,
        "multi_label": True,
        "dataset_version": dataset_version,
        "dataset_path": dataset_path_rel,
        "best_val_macro_f1": round(best_macro_f1, 4),
        "epochs": epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "multi_class_samples": n_multi,
        "class_counts": class_counts,
        "thresholds": thresholds,
        "history": history,
        "model_path": str(onnx_path.relative_to(REPO_ROOT)),
        "backbone_path": str(backbone_path.relative_to(REPO_ROOT)),
        "eval": {"val": eval_val},
    }
    # СНАЧАЛА пишем манифест (модель+пороги+val-метрики) — чтобы падение/OOM на полном
    # eval по всему датасету (тяжёлый на слабом сервере) НЕ потеряло манифест.
    manifest_path = write_classify_manifest(
        model_tag, metrics=metrics, dataset_version=dataset_version,
        dataset_path=dataset_path_rel, multi_label=True, thresholds=thresholds)
    print(f"\nЛучший val macro-F1: {best_macro_f1:.3f}")
    print(f"Модель:   {onnx_path}")
    print(f"Манифест: {manifest_path}", flush=True)

    # Полный eval по всему датасету — best-effort (может съесть память); манифест уже сохранён
    for cls in CLASSES:
        mv = eval_val["by_class"][cls]
        print(f"  {cls}: thr={mv['threshold']:.2f}  val P={mv['precision']:.3f} R={mv['recall']:.3f} "
              f"F1={mv['f1']:.3f} AP={mv['ap']:.3f} (n={mv['support']})", flush=True)
    try:
        full_loader = DataLoader(CropDataset(valid, _val_tf), batch_size=batch_size)
        yf_true, yf_score = _collect_scores(model, full_loader, device)
        eval_full = _multilabel_metrics(yf_true, yf_score, thresholds)
        print(f"  full:  macro_F1={eval_full['macro_f1']:.3f}  mAP={eval_full['mAP']:.3f}  subset_acc={eval_full['subset_accuracy']:.3f}", flush=True)
        metrics["eval"]["full"] = eval_full
        write_classify_manifest(model_tag, metrics=metrics, dataset_version=dataset_version,
                                dataset_path=dataset_path_rel, multi_label=True, thresholds=thresholds)
    except Exception as e:  # noqa: BLE001
        print(f"  [!] Полный eval пропущен ({type(e).__name__}) — манифест уже сохранён с val-метриками", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение классификатора группы (Модель 1, multi-label)")
    ap.add_argument("--data", type=Path, required=True,
                    help="v4-датасет (single/+multi/+labels.json), labels.json или папка-по-классам")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--no-class-weights", dest="class_weights", action="store_false",
                    help="Отключить pos_weight (по умолчанию включён для баланса мультихота)")
    ap.add_argument("--weighted-sampling", action="store_true",
                    help="WeightedRandomSampler: upweight примеров с редкими классами")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / ".output" / "train" / "3_train_groups" / "run")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = train(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        class_weights=args.class_weights,
        weighted_sampling=args.weighted_sampling,
        output_dir=args.output,
    )
    if result:
        print(f"Время: {time.monotonic() - t0:.1f}с", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
