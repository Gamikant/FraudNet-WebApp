from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

def build_autoencoder(input_dim, layer_ratios, n_features, activation='relu'):
    model = Sequential()
    model.add(Dense(int(float(layer_ratios[0]) * n_features), activation=activation, input_shape=(input_dim,)))
    model.add(Dropout(0.1))
    model.add(Dense(int(float(layer_ratios[1]) * n_features), activation=activation))
    model.add(Dense(int(float(layer_ratios[2]) * n_features), activation='linear'))  # Bottleneck layer
    model.add(Dense(int(float(layer_ratios[1]) * n_features), activation=activation))
    model.add(Dense(int(float(layer_ratios[0]) * n_features), activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(input_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(train_data, val_data, model, epochs=50, batch_size=256):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, 
              validation_data=(val_data, val_data), callbacks=[early_stopping], verbose=0)
    return model

def train_final_autoencoder(train_data, val_data, ratios, activation, n_features):
    model = build_autoencoder(train_data.shape[1], ratios, n_features, activation)
    model = train_autoencoder(train_data, val_data, model)
    return model