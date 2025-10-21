from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import sys
import os

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lab import (
    load_mnist_data,
    preprocess_data,
    train_mlp_model,
    train_rf_model,
    train_cnn_model,
    evaluate_models,
    generate_report
)

# Default arguments for the DAG
default_args = {
    'owner': 'Praveen',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'mnist_classification_pipeline',
    default_args=default_args,
    description='Complete MNIST digit classification pipeline with multiple models',
    schedule_interval='@daily',
    catchup=False,
    tags=['mnist', 'classification', 'ml-pipeline'],
)

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_mnist_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

# Model training task group
with TaskGroup('model_training', tooltip='Train multiple models', dag=dag) as training_group:
    train_mlp_task = PythonOperator(
        task_id='train_mlp',
        python_callable=train_mlp_model,
        dag=dag,
    )
    
    train_rf_task = PythonOperator(
        task_id='train_rf',
        python_callable=train_rf_model,
        dag=dag,
    )
    
    train_cnn_task = PythonOperator(
        task_id='train_cnn',
        python_callable=train_cnn_model,
        dag=dag,
    )

evaluate_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag,
)

# Define task dependencies
load_data_task >> preprocess_task >> training_group >> evaluate_task >> report_task