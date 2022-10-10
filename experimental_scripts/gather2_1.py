"""
# Dimensions are
# 0 - stream
# 1 - base_estimator
# 2 - methods
# 3 - chunk
# 4 - metric
"""
import numpy as np
import os
from config21 import *
from strlearn.utils import scores_to_cummean

names = os.listdir("./dd_streams")
names.sort()
ordered_scores_1 = np.zeros((len(names), 2, 7, 199, 1))
ordered_scores_2 = np.zeros((len(names), 1, 7, 199, 1))
ordered_scores_2_rev = np.zeros((len(names), 3, 2, 199, 1))
# print(ordered_scores_1.shape)

for id, name in enumerate(names):
    scores = np.load("results/ex21_1/%s" % name)
    scores = scores_to_cummean(scores)
    mean_scores = np.squeeze(np.mean(scores, axis=1))
    print(mean_scores, scores.shape)
    try:
        os = np.reshape(scores, (2, 7, 199, 1))
        print(np.mean(os, axis=2))
        ordered_scores_1[id] = os
        print(ordered_scores_1[id])
        print(os.shape)
    except:
        print("STILL WAITING")

for id, name in enumerate(names):
    scores = np.load("results/ex22_1/%s" % name)
    scores = scores_to_cummean(scores)
    mean_scores = np.squeeze(np.mean(scores, axis=1))
    print(mean_scores, scores.shape)
    try:
        os = np.reshape(scores, (1, 7, 199, 1))
        print(np.mean(os, axis=2))
        ordered_scores_2[id] = os
        print(ordered_scores_2[id])
        print(os.shape)
    except:
        print("STILL WAITING")
        
for id, name in enumerate(names):
    scores = np.load("results/ex2_1_rev_all/%s" % name)
    scores = scores_to_cummean(scores)
    mean_scores = np.squeeze(np.mean(scores, axis=1))
    print("TUTAJ:")
    print(mean_scores, scores.shape)
    try:
        os = np.reshape(scores, (3, 2, 199, 1))
        print(np.mean(os, axis=2))
        ordered_scores_2_rev[id] = os
        print(ordered_scores_2_rev[id])
        print(os.shape)
    except:
        print("STILL WAITING")

print(ordered_scores_1.shape)
print(ordered_scores_2.shape)
print(ordered_scores_2_rev.shape)

ex2_1 = np.concatenate((ordered_scores_1, ordered_scores_2), axis=1)
ex2_1 = np.concatenate((ex2_1, ordered_scores_2_rev), axis=2)
np.save("results2_1_rev", ex2_1)
