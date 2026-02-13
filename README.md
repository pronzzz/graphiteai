# GraphiteAI

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)

GraphiteAI is an end-to-end web application that converts real images into high-quality, hand-drawn-style pencil sketches using a Generative Adversarial Network (GAN).

## 🚀 Features

- **High-Quality Sketches**: Uses advanced GAN architecture (Pix2Pix/CycleGAN) for realistic transformations.
- **User-Friendly Interface**: Simple web interface for easy image uploads.
- **Local Processing**: Runs entirely on your local machine for privacy.
- **Customizable**: Train on your own datasets (e.g., facades, landscapes).

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Detailed Guide](GUIDE.md)

## Installation

### Prerequisites

- Python 3.8+
- `pip`
- (Optional) GPU with CUDA support for faster inference/training

### Setup

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/pronzzz/graphiteai.git
    cd graphiteai
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure `flask-cors` is installed.*

3. **Download Model Weights**:
    The application will attempt to download pre-trained weights automatically upon first run. Alternatively, you can place your own `weights.pth` file in `backend/model/`.

## Usage

### 1. Run the Backend (Inference Server)

Start the Flask server which handles the model inference:

```bash
cd backend
python app.py
```

The server will start on **<http://localhost:5001>**.

### 2. Access the Frontend

Open the `frontend/index.html` file in your preferred web browser.

No build step is required for the frontend.

## Project Structure

- `backend/`: Flask API and PyTorch model definition.
- `training/`: Training scripts and dataset handling for cGAN.
- `frontend/`: Simple HTML/JS web interface.
- `setup.sh`: Script to help set up the environment (if available).
- `GUIDE.md`: Detailed guide on architecture and advanced usage.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and the code of conduct.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Pranav Dwivedi
