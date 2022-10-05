"""
Comparison sytnethic MLP + BALS
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


np.set_printoptions(suppress=True)
replications = config.replications()
for replication in range(replications):
    # Select streams, methods and metrics from config
    streams = config.streams(config.random_state() + replication)
    print(len(streams))
    for stream in streams:
        print(stream)
        methods = config.methods()

        print(len(methods), "methods")

        ttt = TestThenTrain(metrics=list(config.metrics().values()), verbose=True)
        ttt.process(streams[stream], methods)

        print(ttt.scores.shape)

        np.save("results/ex32/%s" % stream, ttt.scores)
