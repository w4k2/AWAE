"""
Comparison DD streams  GNB, HT, MLP
"""
import numpy as np
import strlearn as sl
import config2_rev_all as config
from sklearn.base import clone
from tabulate import tabulate
import matplotlib.pyplot as plt
from strlearn.ensembles import WAE
from strlearn.evaluators import TestThenTrain
import sys
import os
from strlearn.metrics import balanced_accuracy_score


np.set_printoptions(suppress=True)

names = os.listdir("./dd_streams")
names.sort()
for name in names:
    stream = sl.streams.NPYParser("./dd_streams/%s" % name, chunk_size=250, n_chunks=200)
    print(name)
    methods = config.methods()

    print(len(methods), "methods")

    ttt = TestThenTrain(metrics=[balanced_accuracy_score], verbose=True)
    ttt.process(stream, methods)

    print(ttt.scores.shape)

    np.save("results/ex2_1_rev_all/%s" % name[:-4], ttt.scores)
