import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps4/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('train_test_split')

# Загрузка обработанных данных
processed_data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/processed_emotions.csv'
processed_data = pd.read_csv(processed_data_path)
X = processed_data.drop(['label'], axis=1)
y = processed_data['label']

# Начало сессии MLflow
with mlflow.start_run():
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2003)

    # Логирование параметров разделения
    mlflow.log_param('test_size', 0.2)
    mlflow.log_param('random_state', 2003)
    mlflow.log_param('Number of Training Samples', X_train.shape[0])
    mlflow.log_param('Number of Test Samples', X_test.shape[0])

    # Сохранение обучающих и тестовых данных
    train_data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/train_data.csv'
    test_data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/test_data.csv'
    pd.concat([X_train, y_train], axis=1).to_csv(train_data_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_data_path, index=False)

    # Логирование обучающих и тестовых данных в качестве артефактов
    mlflow.log_artifact(local_path=train_data_path, artifact_path='train_test_split')
    mlflow.log_artifact(local_path=test_data_path, artifact_path='train_test_split')

    mlflow.end_run()
