import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prepare_data import *
from autoencoder import *

def get_feature_importance(model, data, method='reconstruction_error'):
    """
    Calculate feature importance using either reconstruction error or FPI method
    
    Args:
        model: trained autoencoder model
        data: input data
        method: 'reconstruction_error' or 'fpi'
    """
    if method == 're':
        predictions = model.predict(data)
        importance = np.mean((predictions - data) ** 2, axis=0)
    
    elif method == 'fpi':
        baseline_error = np.mean((model.predict(data) - data) ** 2)
        importance = []
        
        for i in range(data.shape[1]):
            permuted_data = data.copy()
            permuted_data[:, i] = np.random.permutation(permuted_data[:, i])
            permuted_error = np.mean((model.predict(permuted_data) - data) ** 2)
            importance.append(permuted_error - baseline_error)
        
        importance = np.array(importance)
    
    return importance

def save_feature_importance(importance_scores, feature_names, method, prefix):
    """
    Save feature importance scores to CSV
    """
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    df = df.sort_values('Importance', ascending=False)
    df.to_csv(f'feature selection/{prefix}_{method}_importance.csv', index=False)

def plot_feature_importance(importance_scores, feature_names, method, prefix):
    """
    Create horizontal bar plot of feature importance scores
    """
    plt.figure(figsize=(12, max(8, len(feature_names)/4)))
    sorted_idx = np.argsort(importance_scores)
    pos = np.arange(len(sorted_idx))
    
    plt.barh(pos, importance_scores[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance Score')
    plt.title(f'{prefix.capitalize()} Feature Importance ({method})')
    plt.tight_layout()
    plt.savefig(f'figures/{prefix}_{method}_importance.png')
    plt.close()

def feature_selection(dev_F, dev_NF, oos_F, oos_NF, method, n_features, feature_names, feature_threshold, ratios=[0.8,0.5,0.2], activation='relu', dropout=0.1, optimizer='adam', loss='mse', epochs=10, batch_size=32):
    # Train on fraud examples
    autoencoder_F = build_autoencoder(dev_F.shape[1], n_features, ratios, activation, dropout, optimizer, loss)
    autoencoder_F, _ = train_autoencoder(dev_F, oos_F, autoencoder_F, epochs, batch_size)
    autoencoder_NF = build_autoencoder(dev_NF.shape[1], n_features, ratios, activation, dropout, optimizer, loss)
    autoencoder_NF, _= train_autoencoder(dev_NF, oos_NF, autoencoder_NF, epochs, batch_size)
    
    # Get importance scores using both methods
    if method == 're':
        importance_F_re = get_feature_importance(autoencoder_F, dev_F, method)
        importance_NF_re = get_feature_importance(autoencoder_NF, dev_NF, method)
    elif method == 'fpi':
        importance_F_fpi = get_feature_importance(autoencoder_F, dev_F, method)
        importance_NF_fpi = get_feature_importance(autoencoder_NF, dev_NF, method)
    
    # Save importance scores and plots for fraud
    if method == 're':
        save_feature_importance(importance_F_re, feature_names, method, 'abnormal')
        save_feature_importance(importance_NF_re, feature_names, method, 'normal')
        plot_feature_importance(importance_F_re, feature_names, method, 'abnormal')
        plot_feature_importance(importance_NF_re, feature_names, method, 'normal')
    elif method == 'fpi':
        save_feature_importance(importance_F_fpi, feature_names, method, 'abnormal')
        save_feature_importance(importance_NF_fpi, feature_names, method, 'normal')
        plot_feature_importance(importance_F_fpi, feature_names, method, 'abnormal')
        plot_feature_importance(importance_NF_fpi, feature_names, method, 'normal')
    
    # Use reconstruction error method for feature selection
    if method == 're':
        features_to_drop = determine_features_to_drop(importance_F_re, importance_NF_re, feature_threshold)
    elif method == 'fpi':
        features_to_drop = determine_features_to_drop(importance_F_fpi, importance_NF_fpi, feature_threshold)
    return features_to_drop

def determine_features_to_drop(importance_F, importance_NF, feature_threshold=0.1):
    top_features_NF = np.argsort(importance_NF)[-int(len(importance_NF) * feature_threshold):]
    bottom_features_F = np.where(importance_F <= 0)[0]
    features_to_drop = np.union1d(top_features_NF, bottom_features_F)
    return features_to_drop

def drop_features(data, features_to_drop, all_features):
    data = pd.DataFrame(data, columns = all_features)
    return data.drop(columns=features_to_drop)