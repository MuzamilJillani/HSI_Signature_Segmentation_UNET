import torch
import torchvision
from dataset import HsiDataset
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    train_bbdir,
    val_dir,
    val_maskdir,
    val_bbdir
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = HsiDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        bb_dir=train_bbdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = HsiDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        bb_dir = val_bbdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    intersection = 0
    union = 0
    iou_score = 0
    model.eval()

    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            z = z.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(z,x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            intersection = np.logical_and(y, preds)
            union = np.logical_or(y, preds)
            iou_score = np.sum(intersection) / np.sum(union)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"IoU score: {iou_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y, z) in enumerate(loader):
        x = x.to(device=device)
        z= z.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(z,x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
