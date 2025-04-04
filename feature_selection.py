import numpy as np
import pandas as pd
from prepare_data import *
from autoencoder import *

def get_feature_importance(model, data):
    predictions = model.predict(data)
    reconstruction_error = np.mean((predictions - data) ** 2, axis=0)
    return reconstruction_error

def feature_selection(dev_F, dev_NF, oos_F, oos_NF, n_features, feature_threshold, ratios, activation):
    # Train on fraud examples
    model_F = build_autoencoder(dev_F.shape[1], ratios, n_features, activation)
    model_F = train_autoencoder(dev_F, oos_F, model_F)
    importance_F = get_feature_importance(model_F, dev_F)
    
    # Train on non-fraud examples
    model_NF = build_autoencoder(dev_NF.shape[1], ratios,n_features, activation)
    model_NF = train_autoencoder(dev_NF, oos_NF, model_NF)
    importance_NF = get_feature_importance(model_NF, dev_NF)
    
    # Determine features to drop
    features_to_drop = determine_features_to_drop(importance_F, importance_NF, feature_threshold)
    return features_to_drop

def determine_features_to_drop(importance_F, importance_NF, feature_threshold):
    top_features_NF = np.argsort(importance_NF)[-int(len(importance_NF) * feature_threshold):]
    bottom_features_F = np.where(importance_F <= 0)[0]
    features_to_drop = np.union1d(top_features_NF, bottom_features_F)
    return features_to_drop

def drop_features(data, features_to_drop, all_features):
    data = pd.DataFrame(data, columns = all_features)
    return data.drop(columns=features_to_drop)