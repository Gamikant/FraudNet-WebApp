from pipeline import pipeline
import json
import itertools
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
from feature_selection import *
from prepare_data import *
from autoencoder import *
from model import *

with open('config.json', 'r') as f:
    config = json.load(f)

def run_tuning():
    # Base hyperparameters (same as tuning=False case)
    base_params = {
        "train_file": config["train_file"],
        "validation_file": config["validation_file"],
        "test_file": config["test_file"],
        "target_column": config["target_column"],
        "drop_features": "yes",
        "train_on": "normal",
        "tuning": True,
        "feature_selection": "re",
        "feature_threshold": 0.1,
        "model_threshold": 0.5,
        "model": "LogisticRegression",
        "cross_validation": 5
    }

    # Default model parameters
    default_model_params = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 200,
        "random_state": 42
    }

    # Initial feature selection with default parameters
    base_params["model_params"] = default_model_params
    base_params["autoencoder"] = {
        "ratios": [0.8, 0.5, 0.2],
        "epochs": 10,
        "batch_size": 32,
        "dropout": 0.1,
        "hidden_activation": "selu",
        "optimizer": "adam",
        "loss": "mse"
    }
    
    # Run initial feature selection
    pipeline(base_params)
    best_f1 = 0.0

    # Stage 1: Tune autoencoder
    autoencoder_grid = config["autoencoder"]
    base_params["drop_features"] = "no"
    best_autoencoder_params = None

    # Generate all combinations of autoencoder parameters
    keys, values = zip(*autoencoder_grid.items())
    for v in tqdm(itertools.product(*values), desc="Tuning Autoencoder"):
        current_ae_params = dict(zip(keys, v))
        base_params["autoencoder"] = current_ae_params
        
        result, _, _, _, _, _, _, _, _, _ = pipeline(base_params)
        if result > best_f1:
            best_f1 = result
            best_autoencoder_params = current_ae_params

    # Stage 2: Tune regression model
    model_grid = config["model_params"]

    # Fix autoencoder parameters to best found
    base_params["autoencoder"] = best_autoencoder_params

    # Generate all combinations of model parameters
    keys, values = zip(*model_grid.items())
    for v in tqdm(itertools.product(*values), desc="Tuning Regression Model"):
        current_model_params = dict(zip(keys, v))
        current_model_params["random_state"] = 42
        base_params["model_params"] = current_model_params
        
        result, best_model, precision, recall, confusion_mat, final_autoencoder_trained, final_encoder_trained, encoded_dev, encoded_oos, encoded_oot = pipeline(base_params)
        if result > best_f1:
            best_f1 = result
            best_model_params = current_model_params
            best_model = best_model
            best_confusion_mat = confusion_mat
            best_precision = precision
            best_recall = recall
            best_final_autoencoder_trained = final_autoencoder_trained
            best_final_encoder_trained = final_encoder_trained
            best_encoded_dev = encoded_dev
            best_encoded_oos = encoded_oos
            best_encoded_oot = encoded_oot
            # The pipeline function will handle saving the best model
        
        base_params["model_params"] = best_model_params
        # save the results of the best ensemble model
        print("Results:")
        print("Best F1 Score:", best_f1)
        print("Best Precision:", best_precision)
        print("Best Recall:", best_recall)
        print("Best Confusion Matrix:", best_confusion_mat)

        sns.heatmap(best_confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Best Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('figures/best_confusion_matrix.png')
        plt.close()

        save_results(base_params,
                    best_encoded_dev, best_encoded_oos, best_encoded_oot, 
                    best_final_autoencoder_trained, best_final_encoder_trained, best_model)


if __name__ == "__main__":
    tuning = True
    if tuning:
        run_tuning()
    else:
        hyperparameters = {
            "train_file": config["train_file"],
            "validation_file": config["validation_file"],
            "test_file": config["test_file"],
            "target_column":config["target_column"],
            "drop_features": "no",
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