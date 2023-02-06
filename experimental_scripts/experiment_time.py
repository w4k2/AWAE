"""
Computing time comparison SS streams GNB
"""
import numpy as np
import strlearn as sl
from sklearn.base import clone
from tabulate import tabulate
import matplotlib.pyplot as plt
from strlearn.ensembles import WAE
from strlearn.evaluators import TestThenTrain
import sys
import os
from strlearn.metrics import balanced_accuracy_score
from strlearn.ensembles import SEA, AWE, WAE, AUE, KUE, ROSE
from csm import OALE
from sklearn.naive_bayes import GaussianNB
from skmultiflow.meta import LearnPPNSEClassifier
import time


np.set_printoptions(suppress=True)

names = os.listdir("./ss_streams")
names.sort()
n_chunks = [194, 85, 186, 170, 200, 200, 399, 170, 173]

results = np.zeros((len(names), 8))

for idx, name in enumerate(names):
    # stream = sl.streams.NPYParser("./ss_streams/%s" % name, chunk_size=250, n_chunks=n_chunks[idx])
    print(name)
    
    methods = [
        SEA(GaussianNB()),
        AWE(base_estimator=GaussianNB()),
        AUE(base_estimator=GaussianNB()),
        LearnPPNSEClassifier(base_estimator=GaussianNB(), n_estimators=10),
        OALE(base_estimator=GaussianNB()),
        WAE(
        GaussianNB(),
        post_pruning=True,
        theta=0.05,
        weight_calculation_method="bell_curve",
        aging_method="constant",
        ),
        KUE(GaussianNB()),
        ROSE(GaussianNB()),
    ]

    print(len(methods), "methods")
    for id, method in enumerate(methods):
        stream = sl.streams.NPYParser("./ss_streams/%s" % name, chunk_size=250, n_chunks=n_chunks[idx])
        start = time.time()
        ttt = TestThenTrain(metrics=[balanced_accuracy_score], verbose=True)
        ttt.process(stream, [method])
        result = time.time()-start
        print(result)
        results[idx, id] = result

np.save("results/experiment_time/times", results)
