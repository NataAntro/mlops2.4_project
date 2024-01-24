from datetime import datetime as dt, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'nataantro',
    'start_date': dt(2022, 12, 1),
    'retries': 1,
    'retry_delays': timedelta(minutes=1),
    'depends_on_past': False
}

with DAG(
    'project_hw_4',
    default_args=default_args,
    description='MlOps2_4_pipeline',
    schedule_interval=None,
    max_active_runs=1,
    tags=['mlops', 'ml_pipeline']
) as dag:
    get_data = BashOperator(
        task_id='get_data',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps4/scripts/get_data.py'
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps4/scripts/preprocess_data.py'
    )

    train_test_split = BashOperator(
        task_id='train_test_split',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps4/scripts/train_test_split.py'
    )
    
    linear_model = BashOperator(
        task_id='linear_model_train_test',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps4/scripts/linear_model_train_test.py'
    )
    
    svm_model = BashOperator(
        task_id='svm_model_train_test',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps4/scripts/svm_model_train_test.py'
    )
    
    randomforest_model = BashOperator(
        task_id='randomforest_model_train_test',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps4/scripts/randomforest_model_train_test.py'
    )
    
    # Установка порядка выполнения задач
    get_data >> preprocess_data >> train_test_split >> linear_model >> svm_model >> randomforest_model
