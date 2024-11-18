import pandas as pd
from math import sqrt
import sys, os

class Describe():

    @classmethod
    def count(cls, lst):
        return len(lst)

    @classmethod
    def mean(cls, lst):
        return sum(lst) / len(lst)

    @classmethod
    def std(cls, lst):
        mean = Describe.mean(lst)
        return sqrt(sum(map(lambda x: (x - mean)**2, lst)) / len(lst))

    @classmethod
    def min(cls, lst):
        return min(lst)

    @classmethod
    def max(cls, lst):
        return max(lst)

    @classmethod
    def quartile(cls, lst, q):
        if q < 1 or q > 3:
            return
        return lst[(len(lst) // 4) * int(q)]

def main():
    if not os.path.exists(sys.argv[1]):
        print(f'"{sys.argv[1]}" no such file or directory.')
        return
    csv = pd.read_csv(sys.argv[1])

    check = False
    data = {}
    for col in csv:
        if not check:
            if col.strip() == "Best Hand":
                check = True
            continue
        data[col] = sorted([float(x) for x in csv[col] if pd.notnull(x)])

    desc = {
        "Count" : [Describe.count(data[i]) for i in data],
        "Mean"  : [Describe.mean(data[i]) for i in data],
        "Std"   : [Describe.std(data[i]) for i in data],
        "Min"   : [Describe.min(data[i]) for i in data],
        "25%"   : [Describe.quartile(data[i], 1) for i in data],
        "50%"   : [Describe.quartile(data[i], 2) for i in data],
        "75%"   : [Describe.quartile(data[i], 3) for i in data],
        "Max"   : [Describe.max(data[i]) for i in data],
    }

    print(" " * 8 + "".join([f"{i[:13]:>14}" for i in data]))
    for i in desc:
        print(f"{i:<8}" + "".join([f"{elem:14.6f}" for elem in desc[i]]))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()