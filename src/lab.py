import os
import pickle
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for storing intermediate results
DATA_PATH = '/tmp/mnist_data'
MODEL_PATH = '/tmp/mnist_models'
RESULTS_PATH = '/tmp/mnist_results'

# Create directories if they don't exist
for path in [DATA_PATH, MODEL_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)


def load_mnist_data(**context):
    """
    Load MNIST dataset from Keras/TensorFlow
    
    Returns:
        str: Status message
    """
    logger.info("Loading MNIST dataset...")
    
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Save raw data
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
    with open(f'{DATA_PATH}/raw_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    # Push metadata to XCom
    context['task_instance'].xcom_push(
        key='data_shape',
        value={
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'image_shape': X_train.shape[1:]
        }
    )
    
    logger.info(f"Data loaded: Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return "Data loading completed"


def preprocess_data(**context):
    """
    Preprocess the MNIST data
    - Normalize pixel values
    - Reshape for ML models
    - Create train/val/test splits
    - Apply StandardScaler
    
    Returns:
        str: Status message
    """
    logger.info("Preprocessing MNIST data...")
    
    # Load raw data
    with open(f'{DATA_PATH}/raw_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Reshape and normalize
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data
    preprocessed_data = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }
    
    with open(f'{DATA_PATH}/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    logger.info("Data preprocessing completed")
    return "Preprocessing completed"


def train_mlp_model(**context):
    """
    Train Multi-Layer Perceptron model
    
    Returns:
        str: Training completion message with validation accuracy
    """
    logger.info("Training MLP model...")
    
    # Load preprocessed data
    with open(f'{DATA_PATH}/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Create and train MLP model
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=20,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train, y_train)
    
    # Validate model
    val_pred = mlp.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    # Save model
    with open(f'{MODEL_PATH}/mlp_model.pkl', 'wb') as f:
        pickle.dump(mlp, f)
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(
        key='mlp_val_accuracy',
        value=val_accuracy
    )
    
    logger.info(f"MLP Model - Validation Accuracy: {val_accuracy:.4f}")
    return f"MLP training completed with validation accuracy: {val_accuracy:.4f}"


def train_rf_model(**context):
    """
    Train Random Forest model
    
    Returns:
        str: Training completion message with validation accuracy
    """
    logger.info("Training Random Forest model...")
    
    # Load preprocessed data
    with open(f'{DATA_PATH}/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Create and train Random Forest model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    # Validate model
    val_pred = rf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    # Save model
    with open(f'{MODEL_PATH}/rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(
        key='rf_val_accuracy',
        value=val_accuracy
    )
    
    logger.info(f"Random Forest Model - Validation Accuracy: {val_accuracy:.4f}")
    return f"RF training completed with validation accuracy: {val_accuracy:.4f}"


def train_cnn_model(**context):
    """
    Train Convolutional Neural Network model using TensorFlow/Keras
    
    Returns:
        str: Training completion message with validation accuracy
    """
    logger.info("Training CNN model...")
    
    # Load raw data for CNN (needs 2D structure)
    with open(f'{DATA_PATH}/raw_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    
    X_train, y_train = raw_data['X_train'], raw_data['y_train']
    X_test, y_test = raw_data['X_test'], raw_data['y_test']
    
    # Normalize and reshape for CNN
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 10)
    
    # Build CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=10,
        batch_size=128,
        validation_data=(X_val, y_val_cat),
        verbose=1
    )
    
    # Get validation accuracy
    val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
    
    # Save model
    model.save(f'{MODEL_PATH}/cnn_model.h5')
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(
        key='cnn_val_accuracy',
        value=val_accuracy
    )
    
    logger.info(f"CNN Model - Validation Accuracy: {val_accuracy:.4f}")
    return f"CNN training completed with validation accuracy: {val_accuracy:.4f}"


def evaluate_models(**context):
    """
    Evaluate all models on test set and select the best one
    
    Returns:
        str: Best model information with test accuracy
    """
    logger.info("Evaluating models on test set...")
    
    # Load test data
    with open(f'{DATA_PATH}/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    results = {}
    
    # Evaluate MLP
    try:
        with open(f'{MODEL_PATH}/mlp_model.pkl', 'rb') as f:
            mlp = pickle.load(f)
        mlp_pred = mlp.predict(X_test)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        results['MLP'] = {
            'accuracy': mlp_accuracy,
            'predictions': mlp_pred,
            'report': classification_report(y_test, mlp_pred, output_dict=True)
        }
        logger.info(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating MLP: {e}")
    
    # Evaluate Random Forest
    try:
        with open(f'{MODEL_PATH}/rf_model.pkl', 'rb') as f:
            rf = pickle.load(f)
        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        results['RandomForest'] = {
            'accuracy': rf_accuracy,
            'predictions': rf_pred,
            'report': classification_report(y_test, rf_pred, output_dict=True)
        }
        logger.info(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating Random Forest: {e}")
    
    # Evaluate CNN
    try:
        with open(f'{DATA_PATH}/raw_data.pkl', 'rb') as f:
            raw_data = pickle.load(f)
        
        X_test_cnn = raw_data['X_test'].astype('float32') / 255.0
        X_test_cnn = X_test_cnn.reshape(-1, 28, 28, 1)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
        
        cnn_model = tf.keras.models.load_model(f'{MODEL_PATH}/cnn_model.h5')
        cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
        cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
        
        results['CNN'] = {
            'accuracy': cnn_accuracy,
            'predictions': cnn_pred,
            'report': classification_report(y_test, cnn_pred, output_dict=True)
        }
        logger.info(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating CNN: {e}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"Best model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
    
    # Save results
    with open(f'{RESULTS_PATH}/evaluation_results.json', 'w') as f:
        json.dump({k: {'accuracy': v['accuracy'], 'report': v['report']} 
                  for k, v in results.items()}, f, indent=2)
    
    return f"Best model: {best_model[0]} with test accuracy: {best_model[1]['accuracy']:.4f}"


def generate_report(**context):
    """
    Generate final report with all results
    
    Returns:
        str: Report generation completion message
    """
    logger.info("Generating final report...")
    
    # Load evaluation results
    with open(f'{RESULTS_PATH}/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Get data shape from XCom
    data_shape = context['task_instance'].xcom_pull(task_ids='load_data', key='data_shape')
    
    # Create report
    report = []
    report.append("="*50)
    report.append("MNIST CLASSIFICATION PIPELINE REPORT")
    report.append("="*50)
    report.append(f"\nExecution Date: {context['execution_date']}")
    report.append(f"\nDataset Information:")
    report.append(f"  - Training samples: {data_shape['train_samples']}")
    report.append(f"  - Test samples: {data_shape['test_samples']}")
    report.append(f"  - Image shape: {data_shape['image_shape']}")
    
    report.append("\nModel Performance on Test Set:")
    report.append("-"*30)
    
    for model_name, model_results in results.items():
        report.append(f"\n{model_name}:")
        report.append(f"  Accuracy: {model_results['accuracy']:.4f}")
        report.append(f"  Precision (weighted): {model_results['report']['weighted avg']['precision']:.4f}")
        report.append(f"  Recall (weighted): {model_results['report']['weighted avg']['recall']:.4f}")
        report.append(f"  F1-Score (weighted): {model_results['report']['weighted avg']['f1-score']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    report.append(f"\n{'='*30}")
    report.append(f"BEST MODEL: {best_model[0]}")
    report.append(f"Test Accuracy: {best_model[1]['accuracy']:.4f}")
    report.append("="*30)
    
    report_text = "\n".join(report)
    
    # Save report
    with open(f'{RESULTS_PATH}/final_report.txt', 'w') as f:
        f.write(report_text)
    
    logger.info("Report generated successfully")
    print(report_text)
    
    return "Report generation completed"
