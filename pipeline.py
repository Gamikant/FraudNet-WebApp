from prepare_data import *
from feature_selection import *
from autoencoder import *
from model import *

def pipeline(hyperparameters):
    dev, oos, oot = load_data(hyperparameters["train_file"], hyperparameters["validation_file"], hyperparameters["test_file"])
    dev_F, dev_NF, oos_F, oos_NF = split_data(dev, oos)
    feature_names = list(dev_F.columns)
    n_features = dev_F.shape[1]
    dev_scaled, oos_scaled, oot_scaled, scaler = standardize_data(dev, oos, oot)
    scaled_dev_F = scaler.transform(dev_F)
    scaled_oos_F = scaler.transform(oos_F)
    scaled_dev_NF = scaler.transform(dev_NF)
    scaled_oos_NF = scaler.transform(oos_NF)
    
    features_to_drop = feature_selection(scaled_dev_F, scaled_dev_NF, scaled_oos_F, scaled_oos_NF, hyperparameters['feature_selection'], n_features, feature_names,
                                         hyperparameters['feature_threshold'], 
                                         hyperparameters['autoencoder']['ratios'], 
                                         hyperparameters['autoencoder']['hidden_activation'],
                                         hyperparameters['autoencoder']['dropout'],
                                         hyperparameters['autoencoder']['optimizer'],
                                         hyperparameters['autoencoder']['loss'],
                                         hyperparameters['autoencoder']['epochs'],
                                         hyperparameters['autoencoder']['batch_size'])
    features_to_drop = [feature_names[i] for i in features_to_drop]
    new_dev_scaled = drop_features(dev_scaled, features_to_drop, feature_names)
    new_scaled_dev_F = drop_features(scaled_dev_F, features_to_drop, feature_names)
    new_scaled_dev_NF = drop_features(scaled_dev_NF, features_to_drop, feature_names)
    new_scaled_oos_F = drop_features(scaled_oos_F, features_to_drop, feature_names)
    new_scaled_oos_NF = drop_features(scaled_oos_NF, features_to_drop, feature_names)
    new_oos_scaled = drop_features(oos_scaled, features_to_drop, feature_names)
    new_oot_scaled = drop_features(oot_scaled, features_to_drop, feature_names)
    
    if hyperparameters['train_on'] == 'normal':
        train_on = new_scaled_dev_NF
        val_on = new_scaled_oos_NF
    elif hyperparameters['train_on'] == 'abnormal':
        train_on = new_scaled_dev_F
        val_on = new_scaled_oos_F
    
    final_autoencoder_untrained = build_autoencoder(train_on.shape[1], n_features, hyperparameters['autoencoder']['ratios'],
                                         hyperparameters['autoencoder']['hidden_activation'], hyperparameters['autoencoder']['dropout'],
                                         hyperparameters['autoencoder']['optimizer'], hyperparameters['autoencoder']['loss'])
    final_autoencoder_trained, final_encoder_trained = train_autoencoder(train_on, val_on, final_autoencoder_untrained, hyperparameters['autoencoder']['epochs'], hyperparameters['autoencoder']['batch_size'])
    encoded_dev = encode_data(final_encoder_trained, new_dev_scaled)
    encoded_oos = encode_data(final_encoder_trained, new_oos_scaled)
    encoded_oot = encode_data(final_encoder_trained, new_oot_scaled)
    
    # Build more models here
    if hyperparameters["tuning"]:
        # Hyperparameter tuning logic here
        best_model, best_f1, precision, recall, confusion_mat = grid_search(encoded_dev, 
                                                         dev[hyperparameters['target_column']], 
                                                         encoded_oos, 
                                                         oot[hyperparameters['target_column']],
                                                         hyperparameters["model"],
                                                         hyperparameters["model_threshold"],
                                                         hyperparameters["cross_validation"])
    else:
        best_model, best_f1, precision, recall, confusion_mat = train_model(encoded_dev, 
                                                             dev[hyperparameters['target_column']], 
                                                             encoded_oos, 
                                                             oos[hyperparameters['target_column']], 
                                                             hyperparameters["model"], 
                                                             hyperparameters["model_params"], 
                                                             hyperparameters["model_threshold"])
    print(f'f1_score = {best_f1}')
    print(f'precision = {precision}')
    print(f'recall = {recall}')
    print(f'confusion_matrix = {confusion_mat}')
    save_results(hyperparameters, best_model.get_params(), features_to_drop, 
                 encoded_dev, encoded_oos, encoded_oot, final_autoencoder_trained, final_encoder_trained, best_model)