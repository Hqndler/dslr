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

    if data["Hogwarts House"].isnull().any():
        print("Wrong dataset")
        return

    fig, axs = plt.subplots(13, 13)

    fig.set_figwidth(10)
    fig.set_figheight(10)

    count = 0
    for course in data:
        if course == "Hogwarts House":
            continue
        c = -1
        for i in data:
            if i == "Hogwarts House":
                continue
            c += 1
            if count == 12:
                axs[count][c].set_xlabel(i, fontsize=4)
            axs[count][c].set_xticks([])
            axs[count][c].set_yticks([])
            if i == course:
                print(f"[{count}][{c}] = hist({i} x {course})")
                axs[count][c].hist([data.loc[data['Hogwarts House'] == house, course] for house in houses], bins=40)
                continue
            for house in houses:
                x = data.loc[data['Hogwarts House'] == house, i]
                y = data.loc[data['Hogwarts House'] == house, course]
                axs[count][c].scatter(x, y, s=0.1, label=house)
            print(f"[{count}][{c}] = scatter({i} x {course})")
        axs[count][0].set_ylabel(course, fontsize=4)
        count += 1
    axs[0][6].legend(bbox_to_anchor=(0., 1.02, 1., 1), loc='lower left', mode='expand')
    plt.savefig("plots/pair_plot.png", dpi=900)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()