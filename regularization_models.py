"""Retained dense baseline/L2/dropout model layouts; see ATTRIBUTION.md."""
from tensorflow import keras
from tensorflow.keras import layers,regularizers

def build_baseline():
    """Large unregularized model — intentionally designed to overfit."""
    model = keras.Sequential([layers.Input(shape=(30,)), layers.Dense(256, activation='relu'), layers.Dense(256, activation='relu'), layers.Dense(128, activation='relu'), layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_regularized_model_1(lam=0.001):
    """
    Same architecture as baseline but with L2 weight regularization.
    L2 adds lambda * sum(W^2) to the loss at each layer.
    This penalizes large weights, keeping the model smoother.
    """
    model = keras.Sequential([layers.Input(shape=(30,)), layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam)), layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lam)), layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam)), layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_regularized_model_2(p=0.4):
    """
    Same architecture as baseline but with Dropout after each hidden layer.
    During training, each neuron is dropped with probability p.
    At test time, all neurons are active (Keras handles this automatically).
    """
    model = keras.Sequential([layers.Input(shape=(30,)), layers.Dense(256, activation='relu'), layers.Dropout(rate=p), layers.Dense(256, activation='relu'), layers.Dropout(rate=p), layers.Dense(128, activation='relu'), layers.Dropout(rate=p), layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
