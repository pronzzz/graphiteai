# GraphiteAI Detailed Guide

## How It Works

GraphiteAI uses a Generative Adversarial Network (GAN) to transform real photos into pencil sketches. Specifically, it employs image-to-image translation techniques.

### Architecture

The core model is based on **Pix2Pix** or **CycleGAN** architectures, which are designed for learning a mapping from input images to output images.

- **Generator**: Takes a real photo as input and generates a corresponding sketch.
- **Discriminator**: Tries to distinguish between real sketches (from the dataset) and fake sketches (produced by the generator).

During training, these two networks compete against each other, leading to a generator that produces increasingly realistic sketches.

## Project Structure

- **`backend/`**: Contains the Flask application (`app.py`) that serves the model. It handles image uploads, runs inference using the pre-trained weights, and returns the generated sketch.
- **`frontend/`**: A simple web interface to interact with the API. It allows users to upload images and view the results.
- **`training/`**: Scripts and modules for training the GAN model. This includes dataset loaders, model definitions, and training loops.
- **`fast_style_transfer/`**: (If present) Legacy or alternative style transfer implementations.

## Training Your Own Model

If you wish to train the model yourself or use a different dataset:

1. **Prepare Dataset**: You need a dataset of paired images (Photo <-> Sketch). The `facades` dataset is a common starting point.
2. **Environment**: Ensure you have a GPU-enabled environment (CUDA) for faster training, although it can run on CPU (very slowly).
3. **Run Training**:
    Navigate to the `training/` directory and run the training script (check `training/train.py` or similar for arguments).

    ```bash
    cd training
    python train.py --dataroot ./datasets/facades --name run_name --model cycle_gan
    ```

## Backend API

The backend exposes a simple REST API:

- **`POST /transform`** (or similar endpoint):
  - **Input**: Multipart form data with an image file.
  - **Output**: The processed sketch image.

## Frontend Development

The frontend is a static HTML/JS/CSS site. You can modify it directly in the `frontend/` folder. No complex build process is required unless you expand it to use a framework like React or Vue.
