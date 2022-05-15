import numpy as np

def one_hot_encoder(data):
    encoded = np.array([[1 if label == i else 0 for i in range(10)] for label in data])
    return encoded

def one_hot_decoder(data):
    decoded = np.argmax(data, axis = 1)
    return decoded
