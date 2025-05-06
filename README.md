# Docker MNIST FastAPI

This project demonstrates a Convolutional Neural Network (CNN) trained on the MNIST dataset, served via a FastAPI application. The project is fully Dockerized, so you can build, train, and deploy the API with minimal setup.

## Project Overview

- **Training Script**: Trains a CNN on the MNIST dataset and saves model checkpoints.
- **Inference Module**: Loads the trained model and processes image uploads to predict handwritten digits.
- **FastAPI App**: Provides a web interface to upload an image and view the prediction.
- **Docker Support**: Includes a Dockerfile and docker-compose configuration to simplify deployment.

## Project Structure

```
docker_mnist_fastapi/
├── app/
│   ├── __init__.py                # Marks the app as a Python package
│   ├── fastapi_app.py             # FastAPI application file
│   ├── inference.py               # Model inference module
│   ├── training.py                # Training script for the CNN model
│   ├── model_epoch_5.pth          # Sample checkpoint model
│   ├── model_epoch_10.pth         # Sample checkpoint model
│   └── model_final.pth            # Final trained model
├── data/                          # MNIST dataset files
│   └── MNIST/
│       └── raw/                   # Original MNIST raw files
├── templates/
│   └── index.html                 # HTML template for the FastAPI UI
├── .ignore                        # Files and directories to ignore
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Dockerfile for building the container
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
```

## Requirements

- Python 3.8 or later
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://pypi.org/project/Pillow/)
- [Docker](https://docs.docker.com/get-docker/) *(optional, for containerized deployment)*

## Setup and Installation

### Local Setup

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd docker_mnist_fastapi
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup

If you prefer to use Docker for easier setup:

1. **Build the Docker Image:**

   ```bash
   docker build -t mnist-fastapi .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8000:8000 mnist-fastapi
   ```

Alternatively, using Docker Compose:

```bash
docker-compose up --build
```

This command builds the images for both the FastAPI app and the training service, starts the container(s), and maps port 8000 to your host.

## Training the Model

To train the CNN on MNIST locally, run:

```bash
python app/training.py
```

This script will download the MNIST dataset (if not already available), train the CNN, and save checkpoints (e.g., `model_epoch_5.pth`, `model_epoch_10.pth`) along with the final model (`model_final.pth`) in the `app/` directory.

## Running the FastAPI App

Start the FastAPI app locally using Uvicorn:

```bash
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

Then, open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the MNIST digit prediction interface.

## Features

- **Image Upload**: The web interface allows you to upload an image (28x28 or larger will be resized) of a handwritten digit.
- **Model Inference**: The uploaded image is processed by the CNN, and the predicted digit is displayed.
- **Dynamic Checkpoint Loading**: The inference module automatically loads the latest available model checkpoint.

## Troubleshooting

- **Template Not Found**: Ensure that the `templates` folder (with `index.html`) is located in the `app/` directory (or adjust the path in `fastapi_app.py` accordingly).
- **Port Conflicts**: If port `8000` is already in use, specify a different port in the Uvicorn command or free the port.
- **Docker Issues**: Verify that Docker is installed and running. Use Docker Compose for a simplified setup.

## License

*Include your license information here (e.g., MIT License).* 

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

Feel free to update this README with additional information or instructions as needed.