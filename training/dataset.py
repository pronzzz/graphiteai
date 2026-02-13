import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        
        # Check if we have A/B structure
        if os.path.isdir(os.path.join(root, "%s/A" % mode)):
            self.mode = "unaligned"
            self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        else:
            self.mode = "aligned"
            self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        if self.mode == "aligned":
            img = Image.open(self.files[index % len(self.files)])
            w, h = img.size
            img_A = img.crop((0, 0, w // 2, h))
            img_B = img.crop((w // 2, 0, w, h))

            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

            item_A = self.transform(img_A)
            item_B = self.transform(img_B)
            return {"A": item_B, "B": item_A} # Facades: B (photo) -> A (label/sketch). Adjust mapping if needed.
            # Usually Facades is Label -> Photo. We want Photo -> Sketch.
            # Facades: Left=Label, Right=Photo.
            # We want Input=Photo, Target=Label/Sketch.
            # So Input (A) should be Right side. Target (B) should be Left side.
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            return {"A": item_A, "B": item_B}

    def __len__(self):
        if self.mode == "aligned":
            return len(self.files)
        else:
            return max(len(self.files_A), len(self.files_B))
