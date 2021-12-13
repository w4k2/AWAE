import numpy as np
# import config as c
from itertools import combinations
from tabulate import tabulate
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
from tabulate import tabulate

from matplotlib import rcParams

### E1 get end

np.set_printoptions(precision=3)

alpha = 0.05

results = np.load("results3.npy")
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

"""
# Dimensions are
# 0 - stream
# 1 - base_estimator
# 2 - method
# 3 - replication
"""

# Analysis
met = ["SEA", "AWE", "AUE", "NSE", "OALE", "(d) WAE", "(o) WAE"]
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
            if met[k] == "SEA":
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
