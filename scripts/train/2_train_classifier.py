"""Обучение классификатора группы (MobileNetV3-Small → ONNX).

Входные данные:
  - Директория images/ с кропами людей
  - labels.json с разметкой (формат из export_data.py)

Выходные данные:
  - models/classify/v<N>.onnx   — ONNX модель для production
  - training_results.json       — метрики

Требования:
  pip install torch torchvision

Usage:
    python scripts/train/train_classifier.py --data .output/training/export_classify_*.zip
    python scripts/train/train_classifier.py --data /path/to/export_dir --epochs 30
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
CLASSES = ["resident", "courier", "delivery", "utilities", "other"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _load_dataset(data_path: Path) -> tuple[Path, list[dict]]:
    """Распаковывает zip если нужно, возвращает (images_dir, labels)."""
    if data_path.suffix == ".zip":
        extract_dir = data_path.parent / data_path.stem
        with zipfile.ZipFile(data_path) as zf:
            zf.extractall(extract_dir)
        data_path = extract_dir

    labels_file = data_path / "labels.json"
    if not labels_file.is_file():
        raise FileNotFoundError(f"labels.json не найден в {data_path}")

    labels = json.loads(labels_file.read_text(encoding="utf-8"))["labels"]
    images_dir = data_path / "images"
    return images_dir, labels


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
    # Заменяем голову классификатора
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
    output_dir: Path,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset, random_split
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Нужен PyTorch: pip install torch torchvision pillow", file=sys.stderr)
        return {}

    images_dir, labels = _load_dataset(data_path)

    # Фильтруем только существующие файлы и известные классы
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

    # Dataset
    _train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    n_val = max(1, int(len(valid) * val_split))
    n_train = len(valid) - n_val
    train_items, val_items = valid[:n_train], valid[n_train:]

    train_loader = DataLoader(
        CropDataset(train_items, _train_tf), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        CropDataset(val_items, _val_tf), batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model = _build_model(NUM_CLASSES, pretrained=True)
    if model is None:
        return {}
    model = model.to(device)

    # Сначала обучаем только голову, потом всю сеть
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # Разморозим веса с середины обучения
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

    # Загружаем лучшие веса и экспортируем в ONNX
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    model.eval()

    # Определяем следующий номер версии
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
    (output_dir / "best.pt").unlink(missing_ok=True)

    metrics = {
        "best_val_acc": round(best_val_acc, 4),
        "epochs": epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "class_counts": class_counts,
        "history": history,
        "model_path": str(onnx_path),
    }
    (output_dir / "training_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nЛучший val_acc: {best_val_acc:.3f}")
    print(f"Модель: {onnx_path}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение классификатора группы")
    ap.add_argument("--data", type=Path, required=True,
                    help="Путь к zip-архиву или директории с images/ и labels.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / ".output" / "train" / "2_train_classifier" / "run")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = train(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        output_dir=args.output,
    )
    if result:
        print(f"Время: {time.monotonic() - t0:.1f}с")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
