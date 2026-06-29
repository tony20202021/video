"""Обучение идентификатора жителей (Модель 2 — MobileNetV3-Small → ONNX).

Входные данные:
  - Zip-архив или директория с images/ и labels.json (поле person_id).
    Формат соответствует /training/export?model=identify.
  Классы (person_id) определяются автоматически по уникальным значениям person_id.

Выходные данные:
  - .models/identify/v<N>.onnx  — ONNX модель с классами в метаданных (ключ "classes")
  - training_results.json        — метрики и список классов

Инициализация весов (два варианта):
  --backbone  path  — backbone из 2_train_groups (.models/classify/backbone.pt).
                      Загружает только features, создаёт новую голову.
                      Используется при первом обучении Модели 2.
  --init-from path  — полные веса предыдущей Модели 2 (.pt).
                      Загружает всё что совпадает, голову создаёт под новое число классов.
                      Используется при повторном обучении с накопленными данными.

Требования:
  pip install torch torchvision onnx

Usage:
    python scripts/train/5_train_residents.py --data export_identify.zip
    python scripts/train/5_train_residents.py --data export_dir --backbone .models/classify/backbone.pt
    python scripts/train/5_train_residents.py --data export_dir --init-from .models/identify/v1.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    data = json.loads(labels_file.read_text(encoding="utf-8"))
    labels = data["labels"]
    images_dir = data_path / "images"
    return images_dir, labels


def _collect_classes(labels: list[dict]) -> list[str]:
    """Извлекает уникальные person_id в алфавитном порядке."""
    ids = sorted({lb["person_id"] for lb in labels if lb.get("person_id")})
    return ids


def _build_model(num_classes: int):
    """MobileNetV3-Small с головой под num_classes жителей."""
    try:
        import torch
        import torchvision.models as M
    except ImportError:
        print("Нужен PyTorch: pip install torch torchvision", file=sys.stderr)
        return None

    model = M.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    return model


def _init_weights(model, init_source: Path | None, init_type: str, device) -> None:
    """Инициализирует веса модели из backbone или предыдущей версии M2."""
    if init_source is None:
        return
    try:
        import torch
        state = torch.load(str(init_source), map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        print(f"  Инициализация ({init_type}): загружено {loaded} тензоров, "
              f"пропущено {len(missing)}, лишних {len(unexpected)}")
    except Exception as e:
        print(f"  [!] Не удалось загрузить {init_type}: {e}", file=sys.stderr)


def _embed_classes_in_onnx(onnx_path: Path, classes: list[str]) -> None:
    """Добавляет список классов в метаданные ONNX-модели."""
    try:
        import onnx
        model_proto = onnx.load(str(onnx_path))
        prop = model_proto.metadata_props.add()
        prop.key = "classes"
        prop.value = json.dumps(classes, ensure_ascii=False)
        onnx.save(model_proto, str(onnx_path))
    except ImportError:
        # Сохраняем в sidecar-файл если onnx не установлен
        sidecar = onnx_path.with_suffix(".classes.json")
        sidecar.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [!] onnx не установлен. Классы → {sidecar}")
    except Exception as e:
        print(f"  [!] Ошибка записи метаданных ONNX: {e}", file=sys.stderr)


def train(
    data_path: Path,
    *,
    backbone: Path | None = None,
    init_from: Path | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 5e-4,
    val_split: float = 0.2,
    output_dir: Path,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Нужен PyTorch: pip install torch torchvision pillow", file=sys.stderr)
        return {}

    images_dir, labels = _load_dataset(data_path)

    classes = _collect_classes(labels)
    if not classes:
        print("Нет person_id в labels.json.", file=sys.stderr)
        return {}

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Жителей (классов): {num_classes}")
    for cls in classes:
        cnt = sum(1 for lb in labels if lb.get("person_id") == cls)
        print(f"  {cls}: {cnt}")

    valid = [
        lb for lb in labels
        if (images_dir / lb["image"]).is_file()
        and lb.get("person_id") in class_to_idx
    ]
    if not valid:
        print("Нет валидных изображений.", file=sys.stderr)
        return {}

    # Сильная аугментация — разная одежда, освещение
    _train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ResidentDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            img = Image.open(images_dir / item["image"]).convert("RGB")
            return self.transform(img), class_to_idx[item["person_id"]]

    n_val = max(1, int(len(valid) * val_split))
    n_train = len(valid) - n_val
    train_items, val_items = valid[:n_train], valid[n_train:]

    train_loader = DataLoader(
        ResidentDataset(train_items, _train_tf), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ResidentDataset(val_items, _val_tf), batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model = _build_model(num_classes)
    if model is None:
        return {}

    # Инициализация весов
    if backbone is not None:
        _init_weights(model, backbone, "backbone", device)
    elif init_from is not None:
        _init_weights(model, init_from, "init-from", device)

    model = model.to(device)

    # Фаза 1: только голова (если есть инициализированный backbone)
    has_pretrained = backbone is not None or init_from is not None
    if has_pretrained:
        for param in model.features.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # Фаза 2: вся сеть с середины обучения
        if has_pretrained and epoch == epochs // 2 + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            print(f"  epoch {epoch}: разморозка всех весов")

        model.train()
        train_correct, train_total = 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
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
    identify_dir = REPO_ROOT / ".models" / "identify"
    identify_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(identify_dir.glob("v*.onnx"))
    next_v = len(existing) + 1
    onnx_path = identify_dir / f"v{next_v}.onnx"

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Встраиваем список классов в метаданные ONNX
    _embed_classes_in_onnx(onnx_path, classes)
    print(f"Классы в ONNX: {classes}")

    # Сохраняем .pt для следующего цикла (--init-from)
    pt_path = identify_dir / f"v{next_v}.pt"
    torch.save(model.state_dict(), pt_path)

    (output_dir / "best.pt").unlink(missing_ok=True)

    metrics = {
        "best_val_acc": round(best_val_acc, 4),
        "epochs": epochs,
        "train_samples": n_train,
        "val_samples": n_val,
        "classes": classes,
        "num_classes": num_classes,
        "history": history,
        "model_path": str(onnx_path),
        "weights_path": str(pt_path),
        "backbone_used": str(backbone) if backbone else None,
        "init_from_used": str(init_from) if init_from else None,
    }
    (output_dir / "training_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nЛучший val_acc: {best_val_acc:.3f}")
    print(f"Модель: {onnx_path}")
    print(f"Веса:   {pt_path}  (для следующего --init-from)")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение идентификатора жителей (Модель 2)")
    ap.add_argument("--data", type=Path, required=True,
                    help="Zip-архив или директория с images/ и labels.json (поле person_id)")
    ap.add_argument("--backbone",  type=Path, default=None,
                    help="Backbone от Модели 1 (.models/classify/backbone.pt) — первый цикл")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="Веса предыдущей Модели 2 (.models/identify/v<N>.pt) — повторное обучение")
    ap.add_argument("--epochs",    type=int,   default=30)
    ap.add_argument("--batch-size", type=int,  default=32)
    ap.add_argument("--lr",        type=float, default=5e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--output",    type=Path,
                    default=REPO_ROOT / ".output" / "train" / "5_train_residents" / "run")
    args = ap.parse_args()

    if args.backbone and args.init_from:
        print("[!] Укажите либо --backbone, либо --init-from, не оба сразу.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = train(
        args.data,
        backbone=args.backbone,
        init_from=args.init_from,
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
