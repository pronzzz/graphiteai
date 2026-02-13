import urllib.request
import tarfile
import os

# CMP Facade Database (standard for Pix2Pix demo)
DATASET_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
DATA_DIR = "training/data"

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    filename = "facades.tar.gz"
    filepath = os.path.join(DATA_DIR, filename)
    
    print(f"Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, filepath)
    
    print("Extracting...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
        
    print(f"Dataset extracted to {DATA_DIR}/facades")

if __name__ == "__main__":
    download_and_extract()
