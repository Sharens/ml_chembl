from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    'owner': 'sharens',
    'start_date': datetime(2023, 1, 1),
}

with DAG('chembl_data_prep', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    
    clean_and_normalize = SparkSubmitOperator(
        task_id='spark_clean_normalize',
        conn_id='spark_default', # Wymaga skonfigurowania połączenia w UI Airflow
        application='/opt/workspace/spark_etl_job.py',
        total_executor_cores='2',
        executor_cores='1',
        executor_memory='1g',
        driver_memory='1g',
        name='chembl_etl_job',
        verbose=True
    )