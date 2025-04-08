from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

def build_autoencoder(input_dim, layer_ratios=[0.8, 0.5, 0.2], activation='relu', dropout=0.1, optimizer='adam', loss='mse'):
    model = Sequential()
    model.add(Dense(int(float(layer_ratios[0]) * input_dim), activation=activation, input_shape=(input_dim,)))
    model.add(Dropout(dropout))
    model.add(Dense(int(float(layer_ratios[1]) * input_dim), activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(int(float(layer_ratios[2]) * input_dim), activation='linear'))  # Bottleneck layer
    model.add(Dense(int(float(layer_ratios[1]) * input_dim), activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(int(float(layer_ratios[0]) * input_dim), activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(input_dim, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def train_autoencoder(train_data, val_data, autoencoder, epochs=50, batch_size=256):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, 
              validation_data=(val_data, val_data), callbacks=[early_stopping])
    encoder = Sequential(autoencoder.layers[:4])  # Extract encoder part
    return autoencoder, encoder