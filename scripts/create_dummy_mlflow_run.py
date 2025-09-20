import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5001")

X = np.array([[0], [1]])
y = np.array([0, 0])
model = DummyClassifier(strategy="most_frequent")
model.fit(X, y)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")
    print(run.info.run_id)
