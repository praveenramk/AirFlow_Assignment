# MNIST Digit Classification Pipeline with Apache Airflow

## Overview

This project implements an automated machine learning pipeline using Apache Airflow to classify handwritten digits from the MNIST dataset. The pipeline orchestrates the complete ML workflow - from data ingestion to model evaluation - comparing three different classification algorithms (Multi-Layer Perceptron, Random Forest, and Convolutional Neural Network) to identify the best-performing model for digit recognition.

## Key Features

- **Automated Workflow**: End-to-end automation of the ML pipeline using Apache Airflow
- **Parallel Processing**: Simultaneous training of multiple models for efficiency
- **Model Comparison**: Automatic evaluation and selection of the best-performing model
- **Comprehensive Reporting**: Detailed performance metrics and analysis
- **Containerized Deployment**: Docker-based setup for easy reproducibility

## Dataset

### MNIST Database Overview

The MNIST (Modified National Institute of Standards and Technology) database is a large collection of handwritten digits widely used for training and testing in the field of machine learning.

### Dataset Characteristics

- **Total Samples**: 70,000 grayscale images
  - Training Set: 60,000 images
  - Test Set: 10,000 images
- **Image Dimensions**: 28 × 28 pixels
- **Number of Classes**: 10 (digits 0-9)
- **Pixel Values**: 0-255 grayscale intensity values
- **Format**: Each image is centered and size-normalized

### Data Pipeline Processing

The pipeline automatically:

1. Downloads the MNIST dataset from TensorFlow/Keras datasets
2. Splits training data into train (54,000) and validation (6,000) sets
3. Normalizes pixel values to [0, 1] range
4. Reshapes data based on model requirements:
   - Flattened (784D vector) for MLP and Random Forest
   - 2D structure (28×28×1) preserved for CNN
5. Applies StandardScaler for traditional ML models

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11 with WSL2
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 10GB free disk space
- **CPU**: 2+ cores recommended

### Software Requirements

- **Docker Desktop**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Python**: 3.8-3.11 (for local development/testing)
- **Web Browser**: Modern browser for Airflow UI access

## Project Structure

```
.
├── dags/                   # Airflow DAG files
│   └── mnist_dag.py       # Main pipeline DAG
├── src/                   # Source code for ML tasks
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_mlp.py
│   ├── train_random_forest.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── report_generator.py
├── logs/                  # Airflow logs
├── plugins/              # Airflow plugins
├── docker-compose.yaml   # Docker Compose configuration
└── README.md            
```

## Quick Start

### 1. Create Required Directories

```bash
mkdir -p ./dags ./logs ./plugins ./src
```

### 2. Set Correct Permissions (Linux/WSL2)

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
sudo chown -R "$(id -u):$(id -g)" ./dags ./logs ./plugins ./src
```

### 3. Initialize the Environment

```bash
docker compose up airflow-init
```

### 4. Start All Services

```bash
docker compose up -d
```

### 5. Create Admin User

```bash
docker compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 6. Access Airflow Web UI

- **URL**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin`

## How It Works

### Detailed Task Workflow

#### 1. Data Loading Task (`load_mnist_data`)

- Fetches MNIST dataset using TensorFlow's built-in datasets
- Stores raw data as pickle files in `/tmp/mnist_data/`
- Records dataset metadata (shape, sample counts) in Airflow XCom
- **Execution time**: ~10-20 seconds

#### 2. Data Preprocessing Task (`preprocess_data`)

- Normalizes pixel values from [0, 255] to [0, 1]
- Creates train/validation split (90/10 ratio)
- Flattens images to 784D vectors for traditional ML
- Applies StandardScaler for feature normalization
- Saves preprocessed data for model consumption
- **Execution time**: ~5-10 seconds

#### 3. Model Training Tasks (Parallel Execution)

##### Multi-Layer Perceptron (MLP)

- **Architecture**: Input(784) → Dense(128) → Dense(64) → Output(10)
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam with early stopping
- **Training time**: ~1-2 minutes

##### Random Forest Classifier

- **Configuration**: 100 decision trees, max depth of 20
- Parallel processing enabled for faster training
- No feature scaling required (tree-based model)
- **Training time**: ~2-3 minutes

##### Convolutional Neural Network (CNN)

- **Architecture**: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dense(10)
- **Input shape**: Preserves 2D structure (28×28×1)
- **Dropout layers**: 0.5 for regularization
- **Training**: 10 epochs with batch size 128
- **Training time**: ~5-10 minutes

#### 4. Model Evaluation Task (`evaluate_models`)

- Tests all trained models on 10,000 test images
- Calculates comprehensive metrics:
  - Accuracy
  - Precision (weighted average)
  - Recall (weighted average)
  - F1-Score
  - Confusion Matrix
- Compares models and identifies the best performer
- Saves evaluation results in JSON format
- **Execution time**: ~30 seconds

#### 5. Report Generation Task (`generate_report`)

- Aggregates all metrics and results
- Creates human-readable text report
- Includes:
  - Dataset statistics
  - Model-wise performance metrics
  - Best model identification
  - Execution metadata

## Services

- **airflow-webserver**: Web UI (port 8080)
- **airflow-scheduler**: DAG scheduler
- **airflow-worker**: Celery task executor
- **postgres**: Metadata database
- **redis**: Message broker for Celery

## ML Dependencies

Pre-installed packages:
- TensorFlow
- scikit-learn
- NumPy
- pandas

## Running the Pipeline

1. Navigate to the Airflow UI at http://localhost:8080
2. Locate the `mnist_classification_pipeline` DAG
3. Toggle the DAG to "On" state
4. Click "Trigger DAG" to start the pipeline
5. Monitor progress in the Graph or Tree view
6. Check logs for detailed execution information

## Output and Results

### Model Artifacts

Models are saved in `/tmp/mnist_models/`:
- `mlp_model.h5` - Multi-Layer Perceptron
- `random_forest_model.pkl` - Random Forest
- `cnn_model.h5` - Convolutional Neural Network

### Evaluation Results

Results are saved in `/tmp/mnist_results/`:
- `evaluation_results.json` - Detailed metrics for all models
- `final_report.txt` - Human-readable summary report

### Output

```
Model Performance on Test Set:
------------------------------

MLP:
  Accuracy: 0.9751
  Precision (weighted): 0.9751
  Recall (weighted): 0.9751
  F1-Score (weighted): 0.9751

RandomForest:
  Accuracy: 0.9696
  Precision (weighted): 0.9696
  Recall (weighted): 0.9696
  F1-Score (weighted): 0.9696

CNN:
  Accuracy: 0.9923
  Precision (weighted): 0.9923
  Recall (weighted): 0.9923
  F1-Score (weighted): 0.9923

==============================
BEST MODEL: CNN
Test Accuracy: 0.9923
==============================
```

<img width="1899" height="700" alt="image" src="https://github.com/user-attachments/assets/597a7333-ad24-469a-bacc-b56c4eee4b4d" />
