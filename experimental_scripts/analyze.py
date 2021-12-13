import numpy as np
import config1 as c
from itertools import combinations
from tabulate import tabulate
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
import matplotlib.pyplot as plt

from matplotlib import rcParams

alpha = 0.05

results = np.load("results1.npy")

"""
# Dimensions are
# 0 - replication
# 1 - stream
# 2 - base_estimator
# 3 - post_pruning
# 4 - theta
# 5 - weight_calculation_method
# 6 - aging_method
# 7 - chunk
# 8 - metric
"""

# Flatten replication, chunks and metric
results = np.mean(results, axis=(7, 8))
results = np.moveaxis(results, 0, 6)

# 0 - stream
# 1 - base_estimator
# 2 - post_pruning
# 3 - theta
# 4 - weight_calculation_method
# 5 - aging_method
# 6 - replication


clfs = ["GNB", "HT", "MLP"]
strs = ["SUDDEN", "GRADUAL", "INCREMENTAL"]

keys = {
    0: ["post ", " pre "],
    1: ["0%", "2.5%", "5%", "7.5%", "10%"],
    2: ["same ", "kun.", " pta ", "bell "],
    3: ["propo", "const", "gauss"],
}

# Optimas

print(results[0, 0].shape)
mresults = np.mean(results, axis=6)
for j in range(3):
    for i in range(3):
        a = np.max(mresults[i, j])
        b = np.array(np.where(mresults[i, j] == a)).T[0]

        z = np.array(np.where(mresults[i, j] == a)).T

        print(z)

        print(
            "%5s" % clfs[j],
            " & ",
            "%12s" % strs[i],
            " & ",
            "%.3f" % a,
            " & ",
            # b,
            "\\textsc{%s}" % keys[0][b[0]],
            " & ",
            keys[1][b[1]].replace("%", "\\%"),
            " & ",
            "\\textsc{%s}" % keys[2][b[2]],
            " & ",
            "\\textsc{%s}" % keys[3][b[3]],
            " \\\\ ",
        )

to_transpose = [0, 1]
bigtables = [[], [], [], [], [], []]
bigdeps = [[], [], [], [], [], []]

# Iterate classifiers
for clf_idx, clf_results in enumerate(results):
    clf_name = list(c.base_estimators().keys())[clf_idx]

    # Iterate streams
    for s_idx, s_results in enumerate(clf_results):
        parameters = [
            "post_pruning",
            "theta",
            "weight_calculation_method",
            "aging_method",
        ]
        # 0 - post_pruning
        # 1 - theta
        # 2 - weight_calculation_method
        # 3 - aging_method

        # Iterate pairs of parameters
        pairs = combinations(list(range(4)), 2)

        to_transpose = [0, 1]

        pairs = [(1, 0), (2, 0), (0, 3), (1, 2), (1, 3), (2, 3)]

        for pp, pair in enumerate(pairs):
            print(
                "\n[%s-%i] %s vs %s"
                % (clf_name, s_idx, parameters[pair[0]], parameters[pair[1]])
            )
            axis = tuple([x for x in list(range(4)) if x not in pair])

            # Calculate comb results and filtered comb_results
            f_comb_results = np.mean(s_results, axis=axis)
            comb_results = np.mean(f_comb_results, axis=2)

            if pp in to_transpose:
                f_comb_results = np.swapaxes(f_comb_results, 0, 1)
                comb_results = np.swapaxes(comb_results, 0, 1)

            print(f_comb_results.shape, comb_results.shape)

            # Gather best idx
            best_idx = [i[0] for i in np.where(np.max(comb_results) == comb_results)]

            cmp_a = f_comb_results[best_idx[0], best_idx[1]]

            print(best_idx)

            dependencies = np.array(
                [
                    [
                        ttest_ind(cmp_a, f_comb_results[i, j]).pvalue
                        for j, b in enumerate(keys[pair[1]])
                    ]
                    for i, a in enumerate(keys[pair[0]])
                ]
            )
            print(dependencies)

            bigdeps[pp] += [dependencies]
            bigtables[pp] += [comb_results]

            tabres = [
                [keys[pair[0]][y_i]]
                + [
                    "%.3f %s" % (x, "d" if dependencies[y_i][x_i] >= alpha else "")
                    for x_i, x in enumerate(y)
                ]
                + ["%.3f" % np.mean(comb_results, axis=1)[y_i]]
                for y_i, y in enumerate(comb_results)
            ]

            tabres += [
                ["-mean"]
                + ["%.3f" % i for i in np.mean(comb_results, axis=0)]
                + ["%.3f" % np.mean(comb_results)]
            ]

            tab = tabulate(tabres, headers=keys[pair[1]] + ["-mean"])

            print(tab)


print("HERE")

bigtables = [np.array(b) for b in bigtables]
bigtables = [b.reshape((3, 3, b.shape[-2], b.shape[-1])) for b in bigtables]

bigdeps = [np.array(b) for b in bigdeps]
bigdeps = [b.reshape((3, 3, b.shape[-2], b.shape[-1])) for b in bigdeps]


tables = [
    np.concatenate(
        [np.concatenate([z[i, j] for i, b in enumerate(a)]) for j, a in enumerate(z)],
        axis=1,
    )
    for z in bigtables
]


debs = [
    np.concatenate(
        [np.concatenate([z[i, j] for i, b in enumerate(a)]) for j, a in enumerate(z)],
        axis=1,
    )
    for z in bigdeps
]


pairs = combinations(list(range(4)), 2)
for i, pair in enumerate(pairs):
    print("\nTABLE %i" % i, pair, parameters[pair[0]], "vs", parameters[pair[1]])

    t = tables[i]
    d = debs[i] >= alpha

    tadam = [
        ["%s%.3f" % ("GRUBE " if d[j, i] else "", value) for i, value in enumerate(row)]
        for j, row in enumerate(t)
    ]

    grubas = tabulate(tadam, tablefmt="latex_booktabs")
    grubas = grubas.replace("GRUBE", "\\bfseries")

    f = open("tables/table_%s_%s.tex" % (parameters[pair[0]], parameters[pair[1]]), "w")
    f.write(grubas)
    f.close()

    plt.clf()
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 5.5))

    lenj = len(keys[pair[0]])
    lenk = len(keys[pair[1]])

    if i in to_transpose:
        plt.setp(
            ax,
            xticks=range(lenj),
            xticklabels=keys[pair[0]],
            yticks=range(lenk),
            yticklabels=keys[pair[1]],
        )
    else:
        plt.setp(
            ax,
            yticks=range(lenj),
            yticklabels=keys[pair[0]],
            xticks=range(lenk),
            xticklabels=keys[pair[1]],
        )

    for j in range(3):
        for k in range(3):
            if i in to_transpose:
                smap = debs[i][
                    (j * lenk) : ((j + 1) * lenk), (k * lenj) : ((k + 1) * lenj)
                ]
                im = ax[j, k].imshow(smap, cmap="binary_r", aspect="auto")

                # Values
                for l in range(lenk):
                    for m in range(lenj):
                        ax[j, k].text(
                            m,
                            l,
                            "%.2f" % smap[l, m],
                            color="black" if smap[l, m] > 0.5 else "white",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
            else:
                smap = debs[i][
                    (k * lenj) : ((k + 1) * lenj), (j * lenk) : ((j + 1) * lenk)
                ]
                im = ax[k, j].imshow(smap, cmap="binary_r", aspect="auto")
                # Values
                print(smap, smap.shape, lenj, lenk)
                for l in range(lenj):
                    for m in range(lenk):
                        ax[k, j].text(
                            m,
                            l,
                            "%.2f" % smap[l, m],
                            color="black" if smap[l, m] > 0.5 else "white",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
            if j == 2:
                ax[j, k].set_xlabel(clfs[k])
            if k == 0:
                ax[j, k].set_ylabel(strs[j])

    fig.subplots_adjust(top=0.85, left=0.17, right=0.95, bottom=0.1)
    cbar_ax = fig.add_axes([0.17, 0.9, 0.78, 0.025])

    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=[0.05, 1])

    if i in to_transpose:
        fig.suptitle(
            "%s / %s" % (parameters[pair[1]], parameters[pair[0]]), fontsize=12, x=0.57
        )
    else:
        fig.suptitle(
            "%s / %s" % (parameters[pair[0]], parameters[pair[1]]), fontsize=12, x=0.57
        )

    plt.savefig("figures/p%i.png" % i)
    plt.savefig("figures/p%i.eps" % i)
