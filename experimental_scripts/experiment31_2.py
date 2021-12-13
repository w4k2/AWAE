"""
SS Streams AL GNB, HT
"""

import numpy as np
import strlearn as sl
import config31 as config
from sklearn.base import clone
from tabulate import tabulate
import matplotlib.pyplot as plt
from strlearn.ensembles import WAE
from strlearn.evaluators import TestThenTrain
import sys
import multiprocessing
import os
from strlearn.metrics import balanced_accuracy_score


np.set_printoptions(suppress=True)

def worker(stream, name):
    print("Start: %s" % (name[:-4]))
    methods = config.methods()
    ttt = TestThenTrain(metrics=[balanced_accuracy_score], verbose=False)
    ttt.process(stream, methods)
    np.save("results/ex31_2/%s" % name[:-4], ttt.scores)
    print("End: %s" % (name[:-4]))

names = os.listdir("./ss_streams")
names.sort()
n_chunks = [194, 85, 186, 170, 200, 200, 399, 170, 173]
jobs = []
for idx, name in enumerate(names):
    stream = sl.streams.NPYParser("./ss_streams/%s" % name, chunk_size=250, n_chunks=n_chunks[idx])
    p = multiprocessing.Process(target=worker, args=(stream,name,))
    jobs.append(p)
    p.start()
