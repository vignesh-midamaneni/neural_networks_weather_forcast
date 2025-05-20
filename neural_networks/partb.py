# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 01:17:19 2025

@author: vignesh midamaneni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(x):
    x=x-np.mean(x)
    x/=np.std(x)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def backward_propagation(X, y, A1, A2, A3, W1, W2, W3, b1, b2, b3, activation='sigmoid', learning_rate=0.01):
    m = X.shape[0]
    
    dA3 = A3 - y.T
    dZ3 = dA3 * sigmoid(A3) * (1 - sigmoid(A3))
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    dA2 = np.dot(W3.T, dZ3)
    if activation == 'sigmoid':
        dZ2 = dA2 * A2 * (1 - A2)
    elif activation == 'relu':
        dZ2 = dA2 * (A2 > 0)
    elif activation == 'tanh':
        dZ2 = dA2 * (1 - A2**2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(W2.T, dZ2)
    if activation == 'sigmoid':
        dZ1 = dA1 * A1 * (1 - A1)
    elif activation == 'relu':
        dZ1 = dA1 * (A1 > 0)
    elif activation == 'tanh':
        dZ1 = dA1 * (1 - A1**2)
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    
    return W1, b1, W2, b2, W3, b3

def forward_propagation(X, W1, b1, W2, b2, W3, b3, activation='sigmoid'):
    Z1 = np.dot(W1, X.T) + b1
    if activation == 'sigmoid':
        A1 = sigmoid(Z1)
    elif activation == 'relu':
        A1 = relu(Z1)
    elif activation == 'tanh':
        A1 = tanh(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    if activation == 'sigmoid':
        A2 = sigmoid(Z2)
    elif activation == 'relu':
        A2 = relu(Z2)
    elif activation == 'tanh':
        A2 = tanh(Z2)
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    return A1, A2, A3

def custom_train_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15):
    total_size = len(X)
    train_end = int(train_size * total_size)
    val_end = int((train_size + val_size) * total_size)
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train(X, y, hidden1=8, hidden2=4, activation='sigmoid', epochs=1000, learning_rate=0.01):
    m, n = X.shape
    np.random.seed(42)
    W1 = np.random.randn(hidden1, n)
    b1 = np.zeros((hidden1, 1))
    W2 = np.random.randn(hidden2, hidden1)
    b2 = np.zeros((hidden2, 1))
    W3 = np.random.randn(1, hidden2)
    b3 = np.zeros((1, 1))
    
    training_loss = []
    for epoch in range(epochs):
        A1, A2, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3, activation)
        
        loss = cross_entropy_loss(y, A3.T)
        training_loss.append(loss)
        
        W1, b1, W2, b2, W3, b3 = backward_propagation(X, y, A1, A2, A3, W1, W2, W3, b1, b2, b3, activation, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
    
    return W1, b1, W2, b2, W3, b3, training_loss

def evaluate(X, y, W1, b1, W2, b2, W3, b3):
    _, _, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
    loss = cross_entropy_loss(y, A3.T)
    print(f"Loss: {loss}")

def example():
    df = pd.read_csv("weather_forecast_data.csv")
    df['Rain'] = df['Rain'].map({'no rain': 0, 'rain': 1})
    
    X = df.drop('Rain', axis=1).values
    y = df['Rain'].values.reshape(-1, 1)
    
    normalize(X)
    
    X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(X, y)
    
    W1, b1, W2, b2, W3, b3, training_loss = train(X_train, y_train, hidden1=5, hidden2=3, activation='sigmoid', epochs=1000, learning_rate=0.01)
    
    plt.plot(training_loss)
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    print("\nEvaluating on Validation Set:")
    evaluate(X_val, y_val, W1, b1, W2, b2, W3, b3)
    
    print("\nEvaluating on Test Set:")
    evaluate(X_test, y_test, W1, b1, W2, b2, W3, b3)

example()
