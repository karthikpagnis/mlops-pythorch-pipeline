# MLOps PyTorch Pipeline

This project implements an end-to-end ML pipeline for image classification on the CIFAR-10 dataset using PyTorch. It covers local training, Docker containerization, and Kubernetes deployment.

## Project Structure

```
mlops-pytorch-pipeline/
├── src/
│   ├── model.py           # ResNet-18 model (adapted for CIFAR-10)
│   ├── dataset.py         # Data loading and transforms
│   ├── train.py           # Training script
│   └── serve.py           # Flask serving API
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Setup

### Prerequisites

- Python 3.10+
- Docker Desktop
- kubectl
- A Kubernetes cluster (Minikube or kind)

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements/train.txt
```

## Usage

### Training locally

```bash
python src/train.py
```

This reads the config from `configs/training_config.yaml`, trains for the specified number of epochs, and saves the best checkpoint to `checkpoints/`.

### Running the serving API

```bash
pip install -r requirements/serve.txt
python src/serve.py
```

### Docker

```bash
# Training
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v1

# Serving
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints mlops-serve:v1
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# After training is done
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Testing the API

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Model

- Architecture: ResNet-18 (modified for 32x32 input)
- Dataset: CIFAR-10 (10 classes)
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Early stopping based on validation loss
