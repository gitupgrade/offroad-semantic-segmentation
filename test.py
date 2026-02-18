import yaml
import torch
import numpy as np
from tqdm import tqdm

from utils.dataset import get_dataloaders
from models import DeepLabWrapper


def fast_confusion_matrix(pred, target, num_classes):
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    k = (target >= 0) & (target < num_classes)
    inds = num_classes * target[k] + pred[k]
    cm = torch.bincount(
        inds, minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return cm


def miou_from_cm(cm):
    intersection = torch.diag(cm).float()
    union = cm.sum(1) + cm.sum(0) - torch.diag(cm)
    iou = intersection / (union + 1e-7)
    return iou.mean().item(), iou.cpu().numpy()


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_classes = int(config["NUM_MASK_CHANNELS"])
    ckpt_path = config.get("SAVE_MODEL_PATH") or config.get("LOAD_MODEL_PATH")
    if ckpt_path is None:
        raise ValueError("Config must contain SAVE_MODEL_PATH or LOAD_MODEL_PATH")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model via wrapper
    model = DeepLabWrapper(model_path=ckpt_path)
    model.model.to(device)
    model.model.eval()

    loaders = get_dataloaders(config["DATA_PATH"], batch_size=2)
    val_loader = loaders["valid"]

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating val"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model.model(images)["out"]
            preds = torch.argmax(outputs, dim=1)

            cm += fast_confusion_matrix(preds, masks, num_classes)

    mean_iou, per_class_iou = miou_from_cm(cm)

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Validation mean IoU (mIoU): {mean_iou:.4f}")
    print("Per-class IoU:")
    for i, v in enumerate(per_class_iou):
        print(f"  Class {i}: {v:.4f}")


if __name__ == "__main__":
    main()
