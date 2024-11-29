import pandas as pd
import os, sys
from math import log
import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(data, weight):
    prediction = np.zeros((data.shape[0], len(weight)))
    for j, d in enumerate(weight.items()):
        prediction[:,j] = sigmoid(np.dot(data, d[1]))
    final = [list(weight)[i] for i in np.argmax(prediction, axis=1)]
    
    with open('houses.csv', 'w') as file:
        file.write("Index,Hogwarts House\n")
        [file.write(f"{c},{house}\n") for c, house in enumerate(final)]

def normalize_data(data):
    for i in data:
        if i == 'Hogwarts House':
            continue
        tmp = data[i].dropna()
        data[i] = data[i].fillna(sum(tmp) / len(tmp))
        data[i] = (data[i] - min(tmp)) / (max(tmp) - min(tmp))
    return data

def main():
    for i in sys.argv[1:]:
        if not os.path.exists(i):
            print(f'"{i}" no such file or directory.')
            return

    data = pd.read_csv(sys.argv[1])
    with open(sys.argv[2], 'r') as r:
        weight = json.load(r)

    for k, v in weight.items():
        weight[k] = np.array(v)

    data = normalize_data(data.drop(["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1))

    predict(data, weight)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()