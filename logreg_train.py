import pandas as pd
import os, sys
import matplotlib.pyplot as plt

def main():
    if not os.path.exists(sys.argv[1]):
        print(f'"{sys.argv[1]}" no such file or directory.')
        return
    data = pd.read_csv(sys.argv[1])

    data = data.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1)

    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()