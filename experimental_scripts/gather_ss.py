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

names = os.listdir("./ss_streams")
names.sort()

for id, name in enumerate(names):
    scores1 = np.load("results/ex21_2/%s" % name)
    scores1 = scores_to_cummean(scores1)
    scores1 = scores1.reshape(2, 7, scores1.shape[1], 1)
    scores2 = np.load("results/ex22_2/%s" % name)
    scores2 = scores_to_cummean(scores2)
    scores2 = scores2.reshape(1, 7, scores2.shape[1], 1)
    print(scores1.shape)
    print(scores2.shape)
    ex2_2 = np.concatenate((scores1, scores2), axis=0)
    np.save("results/ex2_2/%s"%name[:-4], ex2_2)

for id, name in enumerate(names):
    scores1 = np.load("results/ex31_2/%s" % name)
    scores1 = scores_to_cummean(scores1)
    scores1 = scores1.reshape(2, 7, scores1.shape[1], 1)
    scores2 = np.load("results/ex32_2/%s" % name)
    scores2 = scores_to_cummean(scores2)
    scores2 = scores2.reshape(1, 7, scores2.shape[1], 1)
    print(scores1.shape)
    print(scores2.shape)
    ex2_2 = np.concatenate((scores1, scores2), axis=0)
    np.save("results/ex3_2/%s"%name[:-4], ex2_2)
