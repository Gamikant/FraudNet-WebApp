from prepare_data import *
from feature_selection import *
from autoencoder import *
from model import *

def pipeline(dev_path, oos_path, oot_path, hyperparameters):
    dev, oos, oot = load_data(dev_path, oos_path, oot_path)
    dev_F, dev_NF, oos_F, oos_NF = split_data(dev, oos)
    all_features = dev_F.columns
    n_features = dev_F.shape[1]
    dev_scaled, oos_scaled, oot_scaled, scaler = standardize_data(dev, oos, oot)
    dev_F = scaler.transform(dev_F)
    oos_F = scaler.transform(oos_F)
    dev_NF = scaler.transform(dev_NF)
    oos_NF = scaler.transform(oos_NF)
    
    features_to_drop = feature_selection(dev_F, dev_NF, oos_F, oos_NF, n_features, 
                                         hyperparameters['feature_threshold'], 
                                         hyperparameters['ratios'], 
                                         hyperparameters['activation'])
    features_to_drop = [all_features[i] for i in features_to_drop]
    new_dev = drop_features(dev_scaled, features_to_drop, all_features)
    new_dev_F = drop_features(dev_F, features_to_drop, all_features)
    new_dev_NF = drop_features(dev_NF, features_to_drop, all_features)
    new_oos_F = drop_features(oos_F, features_to_drop, all_features)
    new_oos_NF = drop_features(oos_NF, features_to_drop, all_features)
    new_oos = drop_features(oos_scaled, features_to_drop, all_features)
    new_oot = drop_features(oot_scaled, features_to_drop, all_features)
    
    if hyperparameters['train_on'] == 'abnormal':
        train_on = new_dev_NF
        val_on = new_oos_NF
    else:
        train_on = new_dev_F
        val_on = new_oos_F
    
    final_autoencoder = train_final_autoencoder(train_on, val_on, hyperparameters['ratios'], hyperparameters['activation'], n_features)
    encoded_dev = encode_data(final_autoencoder, new_dev)
    encoded_oos = encode_data(final_autoencoder, new_oos)
    encoded_oot = encode_data(final_autoencoder, new_oot)
    
    best_logistic_model, best_f1, precision, recall = grid_search_logistic(encoded_dev, dev['Class'], encoded_oot, oot['Class'])
    print(f'f1_score = {best_f1}')
    print(f'precision = {precision}')
    print(f'recall = {recall}')
    print(confusion_matrix)
    save_results(hyperparameters, best_logistic_model.get_params(), features_to_drop, 
                 encoded_dev, encoded_oos, encoded_oot, final_autoencoder, best_logistic_model)