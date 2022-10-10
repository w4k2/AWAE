import numpy as np
from itertools import combinations
from tabulate import tabulate
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
from tabulate import tabulate

from matplotlib import rcParams

### E1 get end

np.set_printoptions(precision=3)

alpha = 0.05

results = np.load("gathered/results2.npy")
optimized = np.load("gathered/optimized.npy")

rev_scores = np.load("gathered/results2_rev_all.npy")
print(results.shape)
print(rev_scores.shape)
results = np.concatenate((results, rev_scores), axis=3)
print(results.shape)
# exit()

optimized = np.moveaxis(optimized, 0, 1)

"""
# Dimensions are
# 0 - replication
# 1 - stream
# 2 - base_estimator
# 3 - method
# 4 - chunk
# 5 - metric
"""

# Flatten replication, chunks and metric
results = np.mean(results, axis=(4, 5))
# stream, replication
results = np.moveaxis(results, 0, 3)

# SEA, WAEdef, AWE, AUE, WAEopt
results = np.concatenate((results, optimized[:, :, np.newaxis, :]), axis=2)
# withour d WAE
results = results[:, :, [0,1,2,3,4,5,6,8]]
"""
# Dimensions are
# 0 - stream
# 1 - base_estimator
# 2 - method
# 3 - replication
"""

# Analysis
# met = ["SEA", "AWE", "AUE", "NSE", "OALE", "(d) WAE", "(o) WAE"]
# met = ["SEA", "AWE", "AUE", "NSE", "OALE", "KUE",  "ROSE", "(d) WAE", "(o) WAE"]
met = ["SEA", "AWE", "AUE", "NSE", "OALE", "KUE",  "ROSE", "(o) WAE"]
bases = ["GNB", "HT", "MLP"]

analyze_scores = []
analyze_stat = []

for i, stream in enumerate(results):
    print("Stream", i)
    for j, base_estimator in enumerate(stream):
        print(bases[j])

        # Statistical analysis
        mean_results = np.mean(base_estimator, axis=1)
        best_idx = np.argmax(mean_results)

        a = base_estimator[best_idx]

        full_p = np.array(
            [
                [ttest_ind(a, b).pvalue < alpha for a in base_estimator]
                for b in base_estimator
            ]
        )

        full_T = np.array(
            [
                [ttest_ind(a, b).statistic < 0 for a in base_estimator]
                for b in base_estimator
            ]
        )

        tested = full_p * full_T

        for k, method in enumerate(base_estimator):
            b = base_estimator[k]

            better_then = np.where(tested[k])[0] + 1
            if met[k] == "(o) WAE":
                analyze_scores.append(np.mean(method))
                print(
                    "%.3f" % np.mean(method),
                )
                analyze_stat.append(", ".join(["%i" % z for z in better_then]))
                print(
                    ", ".join(["%i" % z for z in better_then]),
                )

print(tabulate(np.array([analyze_scores]), floatfmt=".3f", tablefmt="latex_booktabs"))
print(tabulate(np.array([analyze_stat]), tablefmt="latex_booktabs"))
