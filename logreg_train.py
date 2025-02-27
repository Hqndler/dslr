import pandas as pd
import os, sys
from math import log
import numpy as np
import json
from argparse import ArgumentParser

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(x, y, m):
    return sum(y * np.log(sigmoid(x)) + (1 - y) * np.log(1 - sigmoid(x))) / -m

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

def stochastic_gradient_descent(data, weight, expected, m, i = 0):
    while i < len(data):
        prediction = sigmoid(np.dot(data.values[i : i + m], np.array(weight)))
        gradient = np.dot(prediction - expected[i : i + m], data.values[i : i + m])
        weight -= lr.lr * gradient
        i += m

def batch(data, weight, expected, m):
    prediction = sigmoid(np.dot(data, np.array(weight)))
    gradient = (np.dot(prediction - expected, data)) / m
    weight -= lr.lr * gradient


def fit(data, method):
    weight = {}
    houses = data['Hogwarts House']
    data = data.drop(['Hogwarts House'], axis=1)

    m = len(houses)
    _, w = data.shape

    if method == "stochastic" or method == "mini":
        func = stochastic_gradient_descent
        lr.epoch = 2 if method == "stochastic" else 20
        m = 1 if method == "stochastic" else 100
    else:
        func = batch

    for house in np.unique(houses):
        weight[house] = np.zeros(w)
        expected = np.where(house == houses, 1, 0)
        for _ in range(lr.epoch):
            func(data, weight[house], expected, m)
    [print(f"{k} = {v}") for k, v in weight.items()]
    
    with open("weight.csv", 'w') as file:
        file.write(f"House,{','.join([i for i in data.keys()])}\n")
        for house in weight:
            file.write(f"{house},{','.join([str(i) for i in weight[house]])}\n")

    with open("weight.json", 'w') as file:
        for i in weight:
            weight[i] = list(weight[i])
        json.dump(weight, file)

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("--csv", required=True, type=str, action="store")
    parser.add_argument("--method", required=False, type=str, 
                        choices=["batch", "mini", "stochastic"], 
                        default="batch", action="store")

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f'"{args.csv}" must be a file')
        sys.exit(1)

    return args.csv, args.method

def main():
    csv, method = parse_arg()
    data = pd.read_csv(csv)

    data = data.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1)
    data = normalize_data(data)

    fit(data, method)

if __name__ == "__main__":
    lr = LogisticRegression()
    main()