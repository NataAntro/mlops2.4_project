import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
import os

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps4/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('data_preprocessing')

# Загрузка данных
data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/emotions.csv'
eeg_emotions_data = pd.read_csv(data_path)

# Начало сессии MLflow
with mlflow.start_run():
    # Подготовка данных для модели машинного обучения
    X = eeg_emotions_data.drop(['label'], axis=1)
    y = eeg_emotions_data['label']

    # Логирование размера данных
    mlflow.log_param('Number of Features', X.shape[1])
    mlflow.log_param('Number of Samples', X.shape[0])

    # Кодирование категориальных данных
    labelencoder_emotions = LabelEncoder()
    y = labelencoder_emotions.fit_transform(y)

    # Логирование уникальных меток
    mlflow.log_param('Unique Labels', len(set(y)))

    # Стандартизация признаков в наборе данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Сохранение обработанных данных
    processed_data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/processed_emotions.csv'
    processed_data = pd.concat(
        [pd.DataFrame(X, columns=eeg_emotions_data.drop(['label'], axis=1).columns),
         pd.DataFrame(y, columns=['label'])],
        axis=1
    )
    processed_data.to_csv(processed_data_path, index=False)

    # Логирование обработанных данных в качестве артефакта
    mlflow.log_artifact(local_path=processed_data_path, artifact_path='processed_data')

    mlflow.end_run()
