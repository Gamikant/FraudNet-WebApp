from pipeline import pipeline
import json
import itertools
import numpy as np
import os
from feature_selection import *
from prepare_data import *
from autoencoder import *

with open('config.json', 'r') as f:
    config = json.load(f)

# def run_tuning():

if __name__ == "__main__":
    tuning = False
    if tuning:
        run_tuning()
    else:
        hyperparameters = {
            "train_file": config["train_file"],
            "validation_file": config["validation_file"],
            "test_file": config["test_file"],
            "target_column":config["target_column"],
            "train_on": "normal",
            "tuning": True,
            "feature_selection": "re",
            "feature_threshold": 0.1,
            "model_threshold": 0.5,
            "model": "LogisticRegression",
            "cross_validation": 5
            }
        hyperparameters["tuning"] = False
        hyperparameters["autoencoder"] = {
            "ratios": [0.8, 0.5, 0.2],
            "epochs": 10,
            "batch_size": 32,
            "dropout": 0.1,
            "hidden_activation": "selu",
            "optimizer": "adam",
            "loss": "mse"
        }
        hyperparameters["model_params"] = {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 200,
            "random_state": 42
        }
        pipeline(hyperparameters)