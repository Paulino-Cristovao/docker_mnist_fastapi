# Docker MNIST FastAPI

This project is a demo using a Convolutional Neural Network (CNN) trained on the MNIST dataset, served via a FastAPI application with Docker support. The project includes:

- **Training script**: Train a CNN on MNIST and save checkpoints.
- **Inference module**: Load the model and perform predictions on uploaded images.
- **FastAPI app**: A web interface for users to upload images and see predictions.

## Project Structure

```
docker_mnist_fastapi/
├── app/
│   ├── __init__.py                # Marks the app directory as a p
│   ├── fastapi_app.py             # FastAPI application file
│   ├── inference.py               # Module for model inference
│   └── training.py                # Training script for the CNN model
├── data/                          # Directory for dataset storage (if applicable)
├── .ignore                      # Ignore file for caches, models, and build artifacts
└── README.md                      # Project documentation (this file)
```

## Requirements

- Python 3.8 or later
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Jinja2](https://jinja.palletsprojects.com/)
- Other dependencies as listed in your project's requirements.txt (if available)

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd docker_mnist_fastapi
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *If you do not have a requirements.txt file, install the necessary packages manually.*

## Training the Model

To train the CNN on the MNIST dataset, run the training script:

```bash
python app/training.py
```

This script downloads the MNIST data, trains the CNN, and saves checkpoints as well as the final model.

## Running the FastAPI App

To run the