import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, MLPClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import joblib
import numpy as np

def get_model_and_params(model_type='LogisticRegression'):
    if model_type == 'LogisticRegression':
        model = LogisticRegression()
        param_grid = {'C': [0.1], 'solver': ['lbfgs']}
    elif model_type == 'RandomForest':
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [100], 'max_depth': [None], 'min_samples_split': [2]}
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier()
        param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
    elif model_type == 'XGBoost':
        model = XGBClassifier()
        param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
    elif model_type == 'LightGBM':
        model = LGBMClassifier()
        param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
    elif model_type == 'CatBoost':
        model = CatBoostClassifier()
        param_grid = {'iterations': [100], 'learning_rate': [0.1], 'depth': [3]}
    elif model_type == 'MLP':
        model = MLPClassifier()
        param_grid = {'hidden_layer_sizes': [(100,)], 'activation': ['relu'], 'solver': ['adam']}
    elif model_type == 'SVC':
        model = SVC(probability=True)
        param_grid = {'C': [1], 'kernel': ['rbf']}
    elif model_type == 'KNeighbors':
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [5], 'weights': ['uniform']}
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier()
        param_grid = {'max_depth': [None], 'min_samples_split': [2]}
    elif model_type == 'ExtraTrees':
        model = ExtraTreesClassifier()
        param_grid = {'n_estimators': [100], 'max_depth': [None], 'min_samples_split': [2]}
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Add other models and their parameters here
    return model, param_grid

def grid_search(X_train, y_train, X_test, y_test, model_type='LogisticRegression', threshold=0.5, cv=5):

    percentile = (1-y_train.mean()) * 100

    model, param_grid = get_model_and_params(model_type)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    if threshold:
        threshold = threshold
    else:
        threshold = np.percentile(best_model.predict_proba(X_test)[:, 1], percentile)
    
    # Ensure predictions are discrete class labels
    if hasattr(best_model, 'predict_proba'):
        y_pred = (best_model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    else:
        y_pred = best_model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    return best_model, f1, precision, recall, confusion_mat

def train_model(X_train, y_train, X_test, y_test, model_type, params, threshold):

    percentile = (1-y_train.mean()) * 100

    model = get_model_and_params(model_type)[0]
    model.set_params(**params)
    model.fit(X_train, y_train)
    
    if threshold:
        threshold = threshold
    else:
        threshold = np.percentile(model.predict_proba(X_test)[:, 1], percentile)
    
    # Ensure predictions are discrete class labels
    if hasattr(model, 'predict_proba'):
        y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    return model, f1, precision, recall, confusion_mat

def save_results(log_file, best_params, features_dropped, encoded_dev, encoded_oos, encoded_oot, autoencoder_model, encoder_model, logistic_model):
    pd.DataFrame(log_file).to_csv('pipeline_log.csv', index=False)
    with open('saved best models/best_params.txt', 'w') as f:
        f.write(str(best_params))
    with open('feature selection/features_dropped.txt', 'w') as f:
        f.write(str(features_dropped))
    pd.DataFrame(encoded_dev).to_csv('encoded data/encoded_dev.csv', index=False)
    pd.DataFrame(encoded_oos).to_csv('encoded data/encoded_oos.csv', index=False)
    pd.DataFrame(encoded_oot).to_csv('encoded data/encoded_oot.csv', index=False)
    encoder_model.save('saved best models/encoder_model.h5')
    autoencoder_model.save('saved best models/autoencoder_model.h5')
    joblib.dump(logistic_model, 'saved best models/logistic_model.pkl')