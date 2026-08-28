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
│   ├── persistent-volume-claims.yaml
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

## Architecture

```mermaid
flowchart LR
    A[training_config.yaml] --> B[Docker Training Image]
    B --> C[Kubernetes Training Job]
    C --> D[Checkpoint PVC]
    D --> E[Model Serving Deployment]
    E --> F[ClusterIP Service]
    F --> G[/health and /predict]
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
docker build -f docker/Dockerfile.train -t mlops-train:v2 .
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    mlops-train:v2

# Serving
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    mlops-serve:v1
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volume-claims.yaml
kubectl apply -f k8s/training-job.yaml

# After training is done
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Testing the API

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@/path/to/test_image.png"
```

For Kubernetes testing, run the following command and use the API commands from
another terminal:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

The demonstration configuration uses five epochs, 5,000 training samples, and 1,000 validation
samples. Remove the sample limits to train on the full CIFAR-10 datasets.

## Validation checklist

```bash
kubectl get job,pods,pvc -n ml-training
kubectl logs job/model-training -n ml-training
kubectl get deployment,service,hpa -n ml-training
kubectl describe deployment model-serving -n ml-training
```

Successful validation includes a completed training Job (`1/1`), bound PVCs,
two serving Pods (`1/1 Running`), a healthy `/health` response, and a `/predict`
response containing a class, confidence, and probabilities. Terminal output and
screenshots are attached to pull request descriptions rather than committed.

## Troubleshooting

Docker Desktop Kubernetes must be enabled before applying the manifests. Build
the local images because the manifests use `imagePullPolicy: IfNotPresent`:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v2 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

If a Pod shows `ImagePullBackOff`, build the matching image tag and recreate the
Job or restart the Deployment. If the HPA reports `cpu: <unknown>`, enable the
cluster Metrics Server.

## Reflection

The most challenging part of this project was connecting a training workflow to
an operational serving workflow. The model itself was straightforward, but the
same checkpoint had to move from a training process into persistent Kubernetes
storage and then be loaded by multiple serving replicas. This made image tags,
volume mounts, and startup behavior important details rather than minor
configuration choices.

The first CPU-based Kubernetes run exposed a practical issue: full CIFAR-10
ResNet-18 training can take much longer than expected on a local Docker Desktop
cluster. I addressed this by making dataset limits configurable and using a
five-epoch demonstration with 5,000 training and 1,000 validation samples. The
full dataset remains available by removing those limits. Setting DataLoader
workers to zero also made the local demonstration more reliable in Kubernetes.

The project taught me that reproducibility depends on more than pinning Python
packages. It also requires explicit configuration, predictable checkpoint paths,
health probes, and clear validation commands. The most useful evidence was the
Kubernetes Job completion output, followed by the serving health and prediction
responses. Separating training and serving images kept the runtime focused. In a
production system, I would next add an image registry, authenticated secrets,
experiment tracking, and automated model promotion between environments.

Another important lesson was the value of incremental validation. I tested the
model before containerizing it, checked the Docker images before creating the
Kubernetes Job, and verified the health endpoint before sending a prediction
request. This made failures easier to locate because each stage had a clear
expected result. The CI workflow also provided a repeatable check for syntax,
tests, and image builds instead of relying only on my laptop. Splitting the work
into feature branches and pull requests made the history easier to review and
connected each implementation step with its evidence. I learned that deployment
work is successful only when the software, configuration, storage, and
verification process all agree with one another.

## Git workflow

The work was divided into feature branches and merged through pull requests:

1. `feature/repository-model`
2. `feature/docker-training`
3. `feature/k8s-training`
4. `feature/k8s-serving`

The assignment PDF, virtual environments, datasets, checkpoints, and screenshots
are excluded from Git or attached directly to pull requests.

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
