from pipeline import pipeline
import json
# mlflow or wandb for sweeping

with open('config.json', 'r') as f:
    config = json.load(f)

tuning  = False

if __name__ == "__main__":
    if not tuning:
        hyperparameters = {
            "train_file": "input data/dev.csv",
            "validation_file": "input data/oos.csv",
            "test_file": "input data/oot.csv",
            "target_column": "Class",
            "train_on": "normal",
            "tuning": tuning,
            "autoencoder": 
            {
                "ratios": [0.8, 0.5, 0.2],
                "epochs": 2,
                "batch_size": 32,
                "dropout": 0.1,
                "hidden_activation": "selu",
                "optimizer": "adam",
                "loss": "mse"
            },
            "feature_selection": "fpi",
            "feature_threshold": 0.1,
            "model_threshold": 0,
            "model": "LogisticRegression",
            "model_params":
            {
                "C": 0.1,
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced"
            },
            "cross_validation": 5
        }
        pipeline(hyperparameters)
    else:
        #sweep config
        pass