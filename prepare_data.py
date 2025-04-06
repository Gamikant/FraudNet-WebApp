import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential

def load_data(dev_path, oos_path, oot_path):
    dev = pd.read_csv(dev_path)
    oos = pd.read_csv(oos_path)
    oot = pd.read_csv(oot_path)
    return dev, oos, oot

def standardize_data(dev, oos, oot):
    scaler = StandardScaler()
    dev_scaled = scaler.fit_transform(dev.drop(['Class'], axis=1))
    oos_scaled = scaler.transform(oos.drop(['Class'], axis=1))
    oot_scaled = scaler.transform(oot.drop(['Class'], axis=1))
    return dev_scaled, oos_scaled, oot_scaled, scaler

def split_data(dev, oos):
    dev_F = dev[dev['Class'] == 1].drop(columns=['Class'])
    dev_NF = dev[dev['Class'] == 0].drop(columns=['Class'])
    oos_F = oos[oos['Class'] == 1].drop(columns=['Class'])
    oos_NF = oos[oos['Class'] == 0].drop(columns=['Class'])
    return dev_F, dev_NF, oos_F, oos_NF

def encode_data(encoder, data):
    encoded_data = encoder.predict(data)
    return encoded_data