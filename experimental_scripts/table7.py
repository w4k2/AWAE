import numpy as np
alpha = 0.05

results = np.load("results1_1.npy")

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
strs = ["sudden", "gradual", "incremental"]

keys = {
    0: ["post", "pre"],
    1: ["0%", "2.5%", "5%", "7.5%", "10%"],
    2: ["same ", "kun.", " pta ", "bell "],
    3: ["propo", "const", "gauss"],
}

# Optimas

print(results[0, 0].shape)
mresults = np.mean(results, axis=6)

optimized_results = np.zeros((3,3,5))

print(results.shape)
for j in range(3):
    for i in range(3):
        a = np.max(mresults[i, j])

        b_ = np.array(np.where(mresults[i, j] == a)).T


        multiple = b_.shape[0] > 1


        b = b_[0]

        ad = [i, j] + list(b)

        folds = results[ad[0], ad[1], ad[2], ad[3], ad[4], ad[5]]\

        optimized_results[j,i] = folds

        print(
            "%5s" % clfs[j],
            " & ",
            "\\textsc{%12s}" % strs[i],
            " & ",
            "%.3f" % a,
            " & ",
            # b,
            ("\\textsc{%s}" % keys[0][b[0]] if not multiple else "---"),
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

print(np.mean(optimized_results, axis=2))
