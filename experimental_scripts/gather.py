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
import numpy as np
from config import *

ordered_scores = np.zeros(
    (
        replications(),
        len(streams(0)),
        len(base_estimators()),
        len(post_prunings()),
        len(thetas()),
        len(weight_calculation_methods()),
        len(aging_methods()),
        199,
        1,
    )
)

print(ordered_scores.shape)

for replication in range(replications()):
    _streams = streams(random_state() + replication)
    for s, stream in enumerate(_streams):
        print(replication, s, stream)

        scores = np.load("results/%s.npy" % stream)
        mean_scores = np.squeeze(np.mean(scores, axis=1))
        os = np.reshape(
            scores,
            (
                len(base_estimators()),
                len(post_prunings()),
                len(thetas()),
                len(weight_calculation_methods()),
                len(aging_methods()),
                199,
                1,
            ),
        )
        ordered_scores[replication, s] = os
        print(os.shape)

np.save("results", ordered_scores)

print(ordered_scores.shape)
