import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import joblib

def grid_search_logistic(X_train, y_train, X_test, y_test):
    param_grid = {'C': [0.1], 'solver': ['lbfgs']}
    lr = LogisticRegression()
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict_proba(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    return best_model, f1, precision, recall

def save_results(log_file, best_params, features_dropped, encoded_dev, encoded_oos, encoded_oot, encoder_model, logistic_model):
    pd.DataFrame(log_file).to_csv('pipeline_log.csv', index=False)
    with open('best_params.txt', 'w') as f:
        f.write(str(best_params))
    with open('features_dropped.txt', 'w') as f:
        f.write(str(features_dropped))
    pd.DataFrame(encoded_dev).to_csv('encoded_dev.csv', index=False)
    pd.DataFrame(encoded_oos).to_csv('encoded_oos.csv', index=False)
    pd.DataFrame(encoded_oot).to_csv('encoded_oot.csv', index=False)
    encoder_model.save('encoder_model.h5')
    joblib.dump(logistic_model, 'logistic_model.pkl')