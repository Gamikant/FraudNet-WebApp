from pipeline import pipeline
import json
# mlflow or wandb for sweeping

with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == "__main__":
    