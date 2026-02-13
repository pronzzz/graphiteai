from huggingface_hub import hf_hub_download
import urllib.request
import os

MODEL_DIR = "backend/model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.pth")

def download_weights(url, output_path):
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

def download_hf_model(repo_id, filename, output_path):
    print(f"Downloading {filename} from Hugging Face repo {repo_id}...")
    try:
        # Download file from HF Hub
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Copy or move to output_path if needed, or just return path
        # For simplicity, we can read bytes and write to output_path or just use the cache path
        # But 'urllib.urlretrieve' logic in previous version implied we want it in a specific place.
        # hf_hub_download returns the local path in cache.
        
        # Let's just copy it to be safe and explicit
        import shutil
        shutil.copy(file_path, output_path)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Failed to download from HF: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Example: Using a placeholder or a known model.
    # There isn't a guaranteed "photo2sketch.pth" on HF for *my exact* architecture.
    # However, I can point to a generic one.
    
    # Let's prompt the user or just download a dummy if not found.
    # For now, I will create a dummy weight file so the app runs, 
    # and print instructions on how to use a real one.
    
    print("NOTE: Real pre-trained weights for 'Photo to Pencil Sketch' are large and architecture-specific.")
    print("To use a Hugging Face model, you need to find a `generator.pth` that matches the U-Net architecture.")
    print("Example usage in code:")
    print("  download_hf_model('username/repo', 'generator.pth', 'backend/model/weights.pth')")
    
    # Create a dummy file if it doesn't exist so the app doesn't crash on start (it has fallback anyway)
    if not os.path.exists(WEIGHTS_PATH):
        print("Creating dummy weights for setup verification...")
        with open(WEIGHTS_PATH, 'wb') as f:
            f.write(b'dummy')
