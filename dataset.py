import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle

class HsiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bb_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bb_dir = bb_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.bb = os.listdir(bb_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        bb_path = os.path.join(self.mask_dir, self.masks[index])
        with open(img_path,"rb") as f_in:
            image = pickle.load(f_in)
        image=image[:,:,90]
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float64)
        mask[mask == 255.0] = 1.0
        bb = np.array(Image.open(mask_path).convert("L"), dtype=np.float64)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask , bb=bb)
            image = augmentations["image"]
            mask = augmentations["mask"]
            bb = augmentations["bb"]

        return image, mask, bb
