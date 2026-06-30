"""Быстрая проверка best.pt без ожидания конца обучения.

Usage:
    python scripts/train/3a_eval_pt.py
    python scripts/train/3a_eval_pt.py --checkpoint .output/train/3_train_groups/run/best.pt
    python scripts/train/3a_eval_pt.py --checkpoint best.pt --data .data/groups/v1 --samples 20
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CLASSES = ["1_resident", "2_delivery", "3_utilities"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_CHECKPOINT = REPO_ROOT / ".output" / "train" / "3_train_groups" / "run" / "best.pt"
DEFAULT_DATA = REPO_ROOT / ".data" / "groups" / "v1"


def load_model(checkpoint: Path, device):
    import torch
    import torchvision.models as M

    model = M.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, len(CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model.to(device)


def predict(model, img_path: Path, device):
    import torch
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)[0]
        probs = torch.softmax(logits, dim=0)
    idx = probs.argmax().item()
    return CLASSES[idx], float(probs[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data",       type=Path, default=DEFAULT_DATA)
    ap.add_argument("--samples",    type=int,  default=15,
                    help="Сколько случайных примеров из каждого класса проверить")
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        print(f"[!] Не найден: {args.checkpoint}", file=sys.stderr)
        return 1

    try:
        import torch
    except ImportError:
        print("Нужен PyTorch: pip install torch torchvision pillow", file=sys.stderr)
        return 1

    device = torch.device("cpu")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data}")
    print()

    model = load_model(args.checkpoint, device)

    total_ok = total_all = 0

    for cls in CLASSES:
        cls_dir = args.data / cls
        if not cls_dir.is_dir():
            print(f"  {cls}: папка не найдена")
            continue
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        sample = random.sample(files, min(args.samples, len(files)))

        ok = 0
        errors: list[str] = []
        for f in sample:
            pred, conf = predict(model, f, device)
            if pred == cls:
                ok += 1
            else:
                errors.append(f"{f.name} → {pred} ({conf:.2f})")

        acc = ok / len(sample) if sample else 0
        total_ok += ok
        total_all += len(sample)
        status = "✓" if acc >= 0.7 else ("~" if acc >= 0.5 else "✗")
        print(f"  {status} {cls:20s}  {ok}/{len(sample)}  acc={acc:.2f}  (всего в датасете: {len(files)})")
        for e in errors[:3]:
            print(f"      {e}")
        if len(errors) > 3:
            print(f"      ... ещё {len(errors)-3} ошибок")

    print()
    overall = total_ok / total_all if total_all else 0
    print(f"  Итого: {total_ok}/{total_all}  acc={overall:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
