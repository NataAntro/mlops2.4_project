import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import mlflow
import os
import pickle
from sklearn.model_selection import train_test_split

def evaluate_model(y_true, y_pred):
    # Функция для оценки модели
    # Построение и возврат матрицы ошибок и отчета о классификации
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    return conf_matrix, class_report

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps4/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('logistic_regression_model')

# Загрузка обучающих и тестовых данных
train_data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/train_data.csv'
test_data_path = '/Users/AntroNataExtyl/Desktop/MineMlOps4/data/test_data.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Начало сессии MLflow
with mlflow.start_run():
    # Создание модели логистической регрессии
    model = LogisticRegression(random_state=2003, multi_class='multinomial', solver='lbfgs', max_iter=1000)

    # Логирование параметров модели
    mlflow.log_param('random_state', 2003)
    mlflow.log_param('multi_class', 'multinomial')
    mlflow.log_param('solver', 'lbfgs')
    mlflow.log_param('max_iter', 1000)

    # Обучение модели
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    conf_matrix, class_report = evaluate_model(y_test, y_pred)

    # Логирование метрик модели
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):  # Убедимся, что это не строка
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f'{label}_{metric_name}', metric_value)

    # Сохранение модели
    model_dir = '/Users/AntroNataExtyl/Desktop/MineMlOps4/models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'linear-model.pkl')
    pickle.dump(model, open(model_path,'wb'))

    # Логирование модели в качестве артефакта
    mlflow.log_artifact(local_path=model_path, artifact_path='models')

    mlflow.end_run()
