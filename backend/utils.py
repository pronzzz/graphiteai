import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def tensor2image(tensor):
    """
    Converts a Tensor (C, H, W) to a numpy array (H, W, C) for visualization.
    Un-normalizes from [-1, 1] to [0, 255].
    """
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8).transpose(1, 2, 0)

def load_image(image_path, size=256):
    """
    Loads an image, resizes it, and returns a normalized tensor.
    """
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def save_image(tensor, path):
    """
    Saves a tensor as an image file.
    """
    image_numpy = tensor2image(tensor)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(path)
