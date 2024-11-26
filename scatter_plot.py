import pandas as pd
import os, sys
import matplotlib.pyplot as plt

def main():
    if not os.path.exists(sys.argv[1]):
        print(f'"{sys.argv[1]}" no such file or directory.')
        return
    data = pd.read_csv(sys.argv[1])

    if not os.path.exists("plots"):
        os.makedirs("plots")

    houses = set(data["Hogwarts House"])
    data = data.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1)

    n = 1
    for course in data:
        if course == "Hogwarts House":
            continue
        fig, axs = plt.subplots(4, 3)
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.figure(n)
        print(f"Calculating scatter plot for {course}...")
        c = 0
        for i in data:
            if i == "Hogwarts House" or i == course:
                continue
            ax = axs[(c - 1) // 3][(c - 1) % 3]
            for house in houses:
                x = data.loc[data['Hogwarts House'] == house, i]
                y = data.loc[data['Hogwarts House'] == house, course]
                ax.scatter(x, y, s=1.5, label=house)
            ax.set_ylabel(i[:32])
            ax.set_xlabel(course[:32])
            ax.set_xticks([])
            ax.set_yticks([])
            c += 1
        axs[0][1].legend(bbox_to_anchor=(0., 1.02, 1., 1), loc='lower left', mode='expand')
        plt.savefig(f"plots/{course}.png", dpi=441)
        n += 1

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Add `dataset_train.csv` in argument for example")