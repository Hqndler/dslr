import pandas as pd
import os, sys
from math import log
import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(x, y, m):
    return sum(y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))) / -m

def normalize_data(data):
    for i in data:
        if i == 'Hogwarts House':
            continue
        tmp = data[i].dropna()
        data[i] = data[i].fillna(sum(tmp) / len(tmp))
        data[i] = (data[i] - min(tmp)) / (max(tmp) - min(tmp))
    return data

class LogisticRegression:
    lr = 0.1
    epoch = 1000

def fit(data):
    weight = {}
    houses = data['Hogwarts House']
    data = data.drop(['Hogwarts House'], axis=1)

    m = len(houses)
    _, w = data.shape

    for house in np.unique(houses):
        print(house)
        weight[house] = np.zeros(w)
        expected = np.where(house == houses, 1, 0)
        for _ in range(lr.epoch):
            prediction = sigmoid(np.dot(data, np.array(weight[house])))
            gradient = (np.dot(prediction - expected, data)) / m
            weight[house] -= lr.lr * gradient
    
    with open("weight.csv", 'w') as file:
        file.write(f"House,{','.join([i for i in data.keys()])}\n")
        for house in weight:
            file.write(f"{house},{','.join([str(i) for i in weight[house]])}\n")

    with open("weight.json", 'w') as file:
        for i in weight:
            weight[i] = list(weight[i])
        json.dump(weight, file)

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