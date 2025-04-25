# ================================
# Standard Libraries
# ================================
import os
import warnings
import time
# ================================
# Third-Party Libraries
# ================================
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio

# Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Machine Learning - PyTorch
import torch
from torch.utils.data import DataLoader

# Experiment Tracking
import mlflow
import mlflow.sklearn

# Progress bar
import tqdm

# ================================
# Custom Libraries
# ================================
from featureExtractor import FeatureExtractor
from data_utils import AudioDataSet, collate_fn, create_balanced_subset

# ================================
# Configurations
# ================================

warnings.filterwarnings("ignore", message=".*__audioread_load.*")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

mlflow.set_tracking_uri("file:../experiments")

# ================================
# Functions
# ================================

def run_experiment(meta_data_path, base_dir, extractor_fn, pipeline, batch_size, experiment_name, subset = False):

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        # Logging Extractor Details
        extractor_name = extractor_fn.__name__
        extractor_doc = extractor_fn.__doc__ or "No description provided."
        mlflow.log_param("extractor_name", extractor_name)
        mlflow.log_param("extractor_description", extractor_doc.strip())

        # Logging Pipeline Steps
        pipeline_steps = [name for name, _ in pipeline.steps]
        mlflow.log_param("pipeline_steps", pipeline_steps)

        # Log Pipeline Parameters
        for name, estimator in pipeline.steps:
            if hasattr(estimator, 'get_params'):
                params = estimator.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(f"{name}__{param_name}", param_value)

        # Load metadata
        meta_data = pd.read_csv(meta_data_path)

        if(subset):
            meta_data = create_balanced_subset(meta_data, 1000)

        # Dataset and DataLoader
        dataset = AudioDataSet(meta_data, base_dir, extractor_fn)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

        # Extract features
        X, y = [], []
        for features, labels in tqdm.tqdm(loader):
            if features is None:
                continue
            X.append(features.numpy())
            y.append(labels.numpy())
        X = np.vstack(X)
        y = np.hstack(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # Train pipeline
        pipeline.fit(X_train, y_train)

        # Evaluation
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        for label, scores in report.items():
            if isinstance(scores, dict):
                mlflow.log_metric(f"{label}_precision", scores["precision"])
                mlflow.log_metric(f"{label}_recall", scores["recall"])
                mlflow.log_metric(f"{label}_f1-score", scores["f1-score"])

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Experiment logged with accuracy: {acc}")


if __name__ == "__main__":

    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    from extractors import extract_mfcc_mean_std_26
    experiment_name = "shehab" + str(time.time())

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='linear', C=1.0, gamma='scale'))
    ])

    run_experiment(
        meta_data_path = "../data/work.csv",
        base_dir="../data/work",
        extractor_fn=extract_mfcc_mean_std_26,
        pipeline=pipe,
        batch_size=100,
        experiment_name=experiment_name,
        subset=True
    )
