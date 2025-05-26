import numpy as np
from utils import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, cross_entropy_loss
import matplotlib.pyplot as plt

def initialize_parameters(layer_sizes):
    parameters = {}
    for l in range(1, len(layer_sizes)):
        parameters[f"W{l}"] = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * np.sqrt(2. / layer_sizes[l-1])
        parameters[f"b{l}"] = np.zeros((1, layer_sizes[l]))
    return parameters

def forward_propagation(X, parameters, activation="relu"):
    cache = {"A0": X}
    L = len(parameters) // 2
    for l in range(1, L + 1):
        Z = np.dot(cache[f"A{l-1}"], parameters[f"W{l}"]) + parameters[f"b{l}"]
        if l == L:
            A = softmax(Z)
        else:
            A = relu(Z) if activation == "relu" else sigmoid(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    return cache

def backward_propagation(y, parameters, cache, activation="relu"):
    grads = {}
    L = len(parameters) // 2
    m = y.shape[0]

    dZ = cache[f"A{L}"] - y
    for l in reversed(range(1, L + 1)):
        A_prev = cache[f"A{l-1}"]
        W = parameters[f"W{l}"]
        grads[f"dW{l}"] = np.dot(A_prev.T, dZ) / m
        grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
        if l > 1:
            Z_prev = cache[f"Z{l-1}"]
            dA_prev = np.dot(dZ, W.T)
            dZ = dA_prev * (relu_derivative(Z_prev) if activation == "relu" else sigmoid_derivative(cache[f"A{l-1}"]))
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

def train(X, y, layer_sizes, epochs=10000, learning_rate=0.05, activation="relu", verbose=True):
    parameters = initialize_parameters(layer_sizes)
    losses = []  # Liste pour stocker la perte à chaque epoch
    for i in range(epochs):
        cache = forward_propagation(X, parameters, activation)
        y_pred = cache[f"A{len(layer_sizes)-1}"]
        loss = cross_entropy_loss(y, y_pred)
        losses.append(loss)
        if verbose and i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")
        grads = backward_propagation(y, parameters, cache, activation)
        parameters = update_parameters(parameters, grads, learning_rate)
    return parameters, losses

def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    probs = cache[f"A{len(parameters)//2}"]
    return np.argmax(probs, axis=1)
