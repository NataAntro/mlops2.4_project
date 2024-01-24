import gdown
import os
import mlflow

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps4/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('download_data')

# URLs для файлов Google Drive
url_emotions = 'https://drive.google.com/uc?id=1lGz5owSxzh8qihDso8IGXzL-_HDkmSNL'
url_features = 'https://drive.google.com/uc?id=1_7iD76n8a9h1hmEuRuA4WbojMWFo2-vT'

# Путь для сохранения файлов
save_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data'

# Путь для сохранения существует, иначе его создание
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Запуск сессии MLflow
with mlflow.start_run():
    mlflow.log_param('URL Emotions', url_emotions)
    mlflow.log_param('URL Features', url_features)
    mlflow.log_param('Save Path', save_path)

    # Скачивание и сохранение файлов
    path_emotions = os.path.join(save_path, 'emotions.csv')
    path_features = os.path.join(save_path, 'features_raw.csv')
    gdown.download(url_emotions, path_emotions, quiet=False)
    gdown.download(url_features, path_features, quiet=False)

    # Логирование файлов как артефактов
    mlflow.log_artifact(local_path=path_emotions, artifact_path='data')
    mlflow.log_artifact(local_path=path_features, artifact_path='data')

    mlflow.end_run()
