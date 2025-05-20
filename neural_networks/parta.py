# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:34:11 2025

@author: vignesh midamaneni
"""

import numpy as np
import pandas as pd
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

def forward_propagation(X, y, hidden1=8, hidden2=4, activation='sigmoid'):
    m, n = X.shape
    if activation == 'sigmoid':
        act_fn = sigmoid
    elif activation == 'relu':
        act_fn = relu
    elif activation == 'tanh':
        act_fn = tanh
    np.random.seed(20)
    W1 = np.random.randn(hidden1, n)
    b1 = np.random.randn(hidden1, 1)
    print("W1:\n", W1)
    print("b1:\n", b1)
    W2 = np.random.randn(hidden2, hidden1)
    b2 = np.random.randn(hidden2, 1)
    print("W2:\n", W2)
    print("b2:\n", b2)
    W3 = np.random.randn(1, hidden2)
    b3 = np.random.randn(1, 1)
    print("W3:\n", W3)
    print("b3:\n", b3)
    Z1 = W1 @ X.T + b1
    A1 = act_fn(Z1)
    print("Z1:\n", Z1)
    print("A1:\n", A1)
    Z2 = W2 @ A1 + b2
    A2 = act_fn(Z2)
    print("Z2:\n", Z2)
    print("A2:\n", A2)
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)
    print("Z3:\n", Z3)
    print("A3:\n", A3)
    y_pred = A3.T
    mse = mse_loss(y, y_pred)
    ce = cross_entropy_loss(y, y_pred)
    return y_pred, mse, ce

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

def example():
    df = pd.read_csv("weather_forecast_data.csv")
    df['Rain'] = df['Rain'].map({'no rain': 0, 'rain': 1})

    X = df.drop('Rain', axis=1).values
    y = df['Rain'].values.reshape(-1, 1)
    X=normalize(X)
    for a in ['sigmoid']:
        print(f"\nUsing Activation Function: {a}")
        y_pred, mse, ce = forward_propagation(X, y, hidden1=5, hidden2=3, activation=a)
        print("Predictions:\n", y_pred)
        print("MSE Loss:", mse)
        print("Cross Entropy Loss:", ce)
example()
