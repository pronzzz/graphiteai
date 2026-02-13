import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import requests
from backend.model.generator import Generator

# Model configuration
MODEL_PATH = "backend/model/weights.pth"
# Placeholder URL - User needs to provide actual if desired, or we rely on fallback
PRETRAINED_URL = "" 

class SketchGenerator:
    def __init__(self, model_path=MODEL_PATH, device="cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = Generator().to(self.device)
        self.model_loaded = False
        
        # Try to load model
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"No model found at {model_path}. Using OpenCV fallback.")

    def opencv_fallback(self, image_pil):
        """
        Generates a sketch using OpenCV filters.
        """
        img_np = np.array(image_pil)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)
        return Image.fromarray(img_blend)

    def predict(self, image_pil):
        if not self.model_loaded:
            return self.opencv_fallback(image_pil)
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Post-process
            # Denormalize to [0, 1] then [0, 255]
            output_tensor = output_tensor.squeeze(0).cpu()
            output_image = (output_tensor * 0.5 + 0.5).clamp(0, 1)
            output_pil = transforms.ToPILImage()(output_image)
            return output_pil

        except Exception as e:
            print(f"Inference error: {e}. Switching to fallback.")
            return self.opencv_fallback(image_pil)

sketch_generator = SketchGenerator()
