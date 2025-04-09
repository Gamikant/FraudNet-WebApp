from pipeline import pipeline
import json
import itertools
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import logging
from feature_selection import *
from prepare_data import *
from autoencoder import *
from model import *

# Configure logging
logging.basicConfig(
    filename='hyperparameter_tuning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open('config.json', 'r') as f:
    config = json.load(f)

def run_tuning(default_hyperparameters):
    logging.info("Starting hyperparameter tuning process")
    # Base hyperparameters (same as tuning=False case)
    base_params = default_hyperparameters.copy()
    logging.info(f"Initial parameters: {base_params}")
    
    # Run initial feature selection
    logging.info("Running initial feature selection using default parameters")
    pipeline(base_params)
    best_f1 = 0.0

    # Stage 1: Tune autoencoder
    logging.info("Starting autoencoder tuning - Stage 1")
    autoencoder_grid = config["autoencoder"]
    base_params["drop_features"] = "no"
    best_autoencoder_params = None

    # Generate all combinations of autoencoder parameters
    keys, values = zip(*autoencoder_grid.items())
    total_combinations = len(list(itertools.product(*values)))
    logging.info(f"Testing {total_combinations} autoencoder parameter combinations")
    
    for v in tqdm(itertools.product(*values), desc="Tuning Autoencoder"):
        current_ae_params = dict(zip(keys, v))
        logging.debug(f"Testing autoencoder parameters: {current_ae_params}")
        base_params["autoencoder"] = current_ae_params
        
        result, _, _, _, _, _, _, _, _, _ = pipeline(base_params)
        if result > best_f1:
            best_f1 = result
            best_autoencoder_params = current_ae_params
            logging.info(f"New best F1 score: {best_f1} with parameters: {current_ae_params}")

    logging.info("Starting regression model tuning - Stage 2")
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
        logging.info("Final Results:")
        logging.info(f"Best F1 Score: {best_f1}")
        logging.info(f"Best Precision: {best_precision}")
        logging.info(f"Best Recall: {best_recall}")
        logging.info(f"Best Confusion Matrix:\n{best_confusion_mat}")

        sns.heatmap(best_confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Best Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('figures/best_confusion_matrix.png')
        plt.close()

        logging.info("Saving final results and models")
        save_results(base_params,
                    best_encoded_dev, best_encoded_oos, best_encoded_oot, 
                    best_final_autoencoder_trained, best_final_encoder_trained, best_model)


if __name__ == "__main__":
    tuning = True

    default_hyperparameters = {
        "train_file": config["train_file"],
        "validation_file": config["validation_file"],
        "test_file": config["test_file"],
        "target_column":config["target_column"],
        "drop_features": "no",
        "train_on": "normal",
        "tuning": False,
        "feature_selection": "fpi",
        "feature_threshold": 0.1,
        "model_threshold": 0.5,
        "model": "LogisticRegression",
        "cross_validation": 5
        }
    default_hyperparameters["autoencoder"] = {
        "ratios": [0.8, 0.5, 0.2],
        "epochs": 10,
        "batch_size": 32,
        "dropout": 0.1,
        "hidden_activation": "selu",
        "optimizer": "adam",
        "loss": "mse"
    }
    default_hyperparameters["model_params"] = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 200,
        "random_state": 42
    }
    if tuning:
        default_hyperparameters["tuning"] = tuning
        default_hyperparameters["drop_features"] = "yes"
        logging.info("Running in tuning mode")
        run_tuning(default_hyperparameters)
    else:
        logging.info("Running in standard mode")
        pipeline(default_hyperparameters)