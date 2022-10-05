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
from config21 import *

ordered_scores = np.zeros((replications(), len(streams(0)), len(base_estimators()), 6, 199, 1))

print(ordered_scores.shape)

for replication in range(replications()):
    _streams = streams(random_state() + replication)
    for s, stream in enumerate(_streams):
        print(replication, s, stream)

        scores = np.load("results/ex21/%s.npy" % stream)
        print(scores.shape)
        
        mean_scores = np.squeeze(np.mean(scores, axis=1))

        print(mean_scores, scores.shape)
        try:
            os = np.reshape(scores, (len(base_estimators()), 6, 199, 1))
            print(np.mean(os, axis=2))
            ordered_scores[replication, s] = os
            print(ordered_scores[replication, s])
            print(os.shape)
        except:
            print("STILL WAITING")


np.save("results21", ordered_scores)

print(ordered_scores)
print(ordered_scores.shape)

# ex21 = np.load('results21.npy')
# ex22 = np.load('results22.npy')
# print(ex21.shape)
# print(ex22.shape)

# ex2 = np.concatenate((ex21, ex22), axis=2)
# print(ex2.shape)
# print(ex2)
# np.save("results2", ex2)
