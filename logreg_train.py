import pandas as pd
import os, sys
import numpy as np
from math import log

class LogisticRegression:
    lr = 0.1
    epoch = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(x, y, m):
    return sum(log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))) / -m

def fit(data):
    houses = data['Hogwarts House']
    data = data.drop(['Hogwarts House'], axis=1)
    weight = {}

    m = len(houses)

    _, w = data.shape

    for house in np.unique(houses):
        expected = np.where(houses == house, 1, 0)
        weight[house] = np.zeros(w)
        print(house)
        for _ in range(lr.epoch):
            predicted = sigmoid(np.dot(data, weight[house].T))
            gradient = np.dot(predicted - expected, data) / m
            weight[house] -= lr.lr * gradient
        print(predicted)
        

def normalize_data(data):
    for i in data:
        if i == 'Hogwarts House':
            continue
        data[i] = data[i].fillna(data[i].mean())
        data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
    return data

def main():
    if not os.path.exists(sys.argv[1]):
        print(f'"{sys.argv[1]}" no such file or directory.')
        return
    data = pd.read_csv(sys.argv[1])

    data = data.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1)
    data = normalize_data(data)

    fit(data)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        lr = LogisticRegression()
        main()