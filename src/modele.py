import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, metrics

import pandas as pd

import random

from copy import deepcopy

def train():
    data = pd.read_csv("../data/Wines.csv")

    X = deepcopy(data)
    X = X.sample(frac = 1)
    print(data)
    print(X)
    Y = deepcopy(X["quality"])

    del X["quality"]
    del X["Id"]

    # Normalisation

    X_train = X[:int(0.8 * len(X))]
    X_test = X[int(0.8 * len(X)):]

    Y_train = Y[:int(0.8 * len(Y))]
    Y_test = Y[int(0.8 * len(Y)):]

    inputs = Input(11)

    h1 = Dense(64, activation="relu")(inputs)
    h2 = Dense(64, activation="relu")(h1)
    h3 = Dense(64, activation="relu")(h2)
    outputs = Dense(10, activation="softmax")(h3)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=100, 
                        validation_data=(X_test, Y_test))
    
    # Save a model
    model.save('../data/model')


def predire(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol):
    # Check if model exists
    try:
        model = tf.keras.models.load_model('../data/model')
    except:
        train()
        model = tf.keras.models.load_model('../data/model')

    prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]])
    i = 0
    indice = 0
    max = 0
    for x in prediction[0]:
        i = i + 1
        if(x > max):
            max = x
            indice = i        
    return indice