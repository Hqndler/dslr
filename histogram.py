import pandas as pd
import os, sys
import matplotlib.pyplot as plt

def main():
    if not os.path.exists(sys.argv[1]):
        print(f'"{sys.argv[1]}" no such file or directory.')
        return
    data = pd.read_csv(sys.argv[1])

    if data["Hogwarts House"].isnull().any():
        print("Wrong dataset")
        return

    _, axs = plt.subplots(5, 3)

    houses = set(data["Hogwarts House"])
    data = data.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1)

    for c, course in enumerate(data):
        if course == "Hogwarts House":
            continue
        if c == 13:
            c = 14
        print(f"{c} = {course} [{(c - 1) // 3}][{(c - 1) % 3}]")
        hist = [data.loc[data['Hogwarts House'] == house, course] for house in houses]
        ax = axs[(c - 1) // 3][(c - 1) % 3]
        ax.hist(hist, density=True, bins=20)
        ax.set_title(course)
        ax.set_xticks([])
        ax.set_yticks([])

    axs[4][0].set_visible(False)
    axs[4][2].set_visible(False)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Add `dataset_train.csv` in argument for example")