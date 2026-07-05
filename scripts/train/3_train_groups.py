"""Обучение классификатора группы (Модель 1 — MobileNetV3-Small → ONNX).

Входные данные:
  - Директория images/ с кропами людей
  - labels.json с разметкой (формат из /training/export)

Выходные данные:
  - .models/classify/v<N>.onnx   — ONNX модель для production
  - .models/classify/backbone.pt — веса backbone для инициализации Модели 2
  - training_results.json        — метрики

Требования:
  pip install torch torchvision

Usage:
    python scripts/train/2_train_groups.py --data .output/training/export_classify_*.zip
    python scripts/train/2_train_groups.py --data /path/to/export_dir --epochs 30
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

from common.utils.classes import GROUP_CLASSES as CLASSES
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _load_dataset(data_path: Path) -> tuple[Path, list[dict]]:
    """Распаковывает zip если нужно, возвращает (images_dir, labels).

    Форматы:
      1. Folder-based (папки = классы, рекомендуется):
           data_path/1_resident/img.jpg  →  class = "1_resident"
      2. Стандартный labels.json (из 1_export_data):
           {"labels": [{"image": "img.jpg", "class": "resident"}]}
      3. Формат 2_label_ui:
           {"labels": {"relative/path.jpg": "class"}}
    """
    if data_path.suffix == ".zip":
        extract_dir = data_path.parent / data_path.stem
        with zipfile.ZipFile(data_path) as zf:
            zf.extractall(extract_dir)
        data_path = extract_dir

    # Folder-based: если есть хотя бы одна подпапка с именем из CLASSES
    if data_path.is_dir():
        class_dirs = [d for d in data_path.iterdir()
                      if d.is_dir() and d.name in CLASS_TO_IDX]
        if class_dirs:
            labels = []
            for cls_dir in sorted(class_dirs):
                for f in sorted(cls_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        labels.append({"image": str(f.relative_to(data_path)), "class": cls_dir.name})
            return data_path, labels

    if data_path.is_file() and data_path.name == "labels.json":
        labels_file = data_path
    else:
        labels_file = data_path / "labels.json"
    if not labels_file.is_file():
        raise FileNotFoundError(
            f"labels.json не найден в {data_path}. "
            f"Передайте папку с подпапками по классам ({', '.join(CLASSES)})"
        )

    raw = json.loads(labels_file.read_text(encoding="utf-8"))
    raw_labels = raw["labels"]

    if isinstance(raw_labels, dict):
        # Формат 2_label_ui: ключ — путь относительно REPO_ROOT
        labels = [{"image": path, "class": cls}
                  for path, cls in raw_labels.items()]
        return REPO_ROOT, labels
    else:
        # Стандартный список: images/ + имена файлов
        return data_path / "images", raw_labels


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


def train(
    data_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    class_weights: bool = False,
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

    images_dir, labels = _load_dataset(data_path)

    valid = [
        lb for lb in labels
        if (images_dir / lb["image"]).is_file()
        and lb["class"] in CLASS_TO_IDX
    ]
    if not valid:
        print("Нет валидных изображений.", file=sys.stderr)
        return {}

    print(f"Изображений: {len(valid)}")
    class_counts = {}
    for lb in valid:
        class_counts[lb["class"]] = class_counts.get(lb["class"], 0) + 1
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    counts_arr = np.array([class_counts.get(c, 0) for c in CLASSES], dtype=np.float32)
    total = counts_arr.sum()

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
            img = Image.open(images_dir / item["image"]).convert("RGB")
            return self.transform(img), CLASS_TO_IDX[item["class"]]

    # Стратифицированное разбиение: val_split % от каждого класса
    by_class: dict[str, list] = {}
    for lb in valid:
        by_class.setdefault(lb["class"], []).append(lb)
    train_items, val_items = [], []
    for cls, items in by_class.items():
        n_v = max(1, int(len(items) * val_split))
        val_items.extend(items[:n_v])
        train_items.extend(items[n_v:])
    n_train, n_val = len(train_items), len(val_items)

    train_ds = CropDataset(train_items, _train_tf)

    if weighted_sampling:
        # каждый класс представлен равномерно в каждом батче
        w_per_class = total / (NUM_CLASSES * np.where(counts_arr > 0, counts_arr, 1))
        sample_weights = [float(w_per_class[CLASS_TO_IDX[lb["class"]]]) for lb in train_items]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_items), replacement=True)
        print(f"WeightedRandomSampler включён")
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(
        CropDataset(val_items, _val_tf), batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model = _build_model(NUM_CLASSES, pretrained=True)
    if model is None:
        return {}
    model = model.to(device)

    # Фаза 1: только голова
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    if class_weights:
        w = torch.tensor(total / (NUM_CLASSES * np.where(counts_arr > 0, counts_arr, 1)),
                         dtype=torch.float32, device=device)
        print(f"class_weights: { {c: round(float(w[i]), 2) for i, c in enumerate(CLASSES)} }")
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # Фаза 2: вся сеть с середины обучения
        if epoch == epochs // 2 + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            print(f"  epoch {epoch}: разморозка всех весов")

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
            train_correct += (out.argmax(1) == targets).sum().item()
            train_total += len(imgs)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == targets).sum().item()
                val_total += len(imgs)

        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0
        scheduler.step()

        history.append({"epoch": epoch, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
        print(f"  epoch {epoch:3d}/{epochs}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")

    # Загружаем лучшие веса
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    model.eval()

    # Экспорт в ONNX
    classify_dir = REPO_ROOT / ".models" / "classify"
    classify_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(classify_dir.glob("v*.onnx"))
    next_v = len(existing) + 1
    onnx_path = classify_dir / f"v{next_v}.onnx"

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Сохраняем backbone для инициализации Модели 2 (5_train_residents.py)
    backbone_path = classify_dir / "backbone.pt"
    backbone_state = {k: v for k, v in model.state_dict().items()
                      if k.startswith("features.")}
    torch.save(backbone_state, backbone_path)
    print(f"Backbone (для Модели 2): {backbone_path}")

    (output_dir / "best.pt").unlink(missing_ok=True)

    metrics = {
        "best_val_acc": round(best_val_acc, 4),
        "epochs": epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "class_counts": class_counts,
        "history": history,
        "model_path": str(onnx_path),
        "backbone_path": str(backbone_path),
    }
    (output_dir / "training_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nЛучший val_acc: {best_val_acc:.3f}")
    print(f"Модель: {onnx_path}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение классификатора группы (Модель 1)")
    ap.add_argument("--data", type=Path, required=True,
                    help="Путь к zip-архиву или директории с images/ и labels.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--class-weights", action="store_true",
                    help="Взвешенная функция потерь (рекомендуется при дисбалансе классов)")
    ap.add_argument("--weighted-sampling", action="store_true",
                    help="WeightedRandomSampler: равномерная выборка классов в каждом батче")
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
        print(f"Время: {time.monotonic() - t0:.1f}с")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
