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
from config3_rev_bals import *

ordered_scores = np.zeros((replications(), len(streams(0)), len(base_estimators()), 2, 199, 1))

print(ordered_scores.shape)

for replication in range(replications()):
    _streams = streams(random_state() + replication)
    for s, stream in enumerate(_streams):
        print(replication, s, stream)

        scores = np.load("results/ex3_rev_bals/%s.npy" % stream)
        print(scores.shape)

        mean_scores = np.squeeze(np.mean(scores, axis=1))

        print(mean_scores, scores.shape)
        try:
            os = np.reshape(scores, (len(base_estimators()), 2, 199, 1))
            print(np.mean(os, axis=2))
            ordered_scores[replication, s] = os
            print(ordered_scores[replication, s])
            print(os.shape)
        except:
            print("STILL WAITING")


np.save("results3_rev_bals", ordered_scores)

print(ordered_scores)
print(ordered_scores.shape)


# ex31 = np.load('results31.npy')
# ex32 = np.load('results32.npy')
# print(ex31.shape)
# print(ex32.shape)

# ex3 = np.concatenate((ex31, ex32), axis=2)
# print(ex3.shape)
# print(ex3)
# np.save("results3", ex3)
