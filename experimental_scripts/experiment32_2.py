"""
SS Streams AL MLP
"""
import numpy as np
import strlearn as sl
import config32 as config
from sklearn.base import clone
from tabulate import tabulate
import matplotlib.pyplot as plt
from strlearn.ensembles import WAE
from strlearn.evaluators import TestThenTrain
import sys
import os
from strlearn.metrics import balanced_accuracy_score


np.set_printoptions(suppress=True)

names = os.listdir("./ss_streams")
names.sort()
n_chunks = [194, 85, 186, 170, 200, 200, 399, 170, 173]

for idx, name in enumerate(names):
    stream = sl.streams.NPYParser("./ss_streams/%s" % name, chunk_size=250, n_chunks=n_chunks[idx])
    print(name)
    methods = config.methods()

    print(len(methods), "methods")

    ttt = TestThenTrain(metrics=[balanced_accuracy_score], verbose=True)
    ttt.process(stream, methods)

    print(ttt.scores.shape)

    np.save("results/ex32_2/%s" % name[:-4], ttt.scores)
