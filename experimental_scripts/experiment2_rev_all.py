"""
Comparison sytnethic GNB, HT, MLP, revision
"""
import numpy as np
import strlearn as sl
import config2_rev_all as config
from strlearn.evaluators import TestThenTrain
import multiprocessing


np.set_printoptions(suppress=True)
replications = config.replications()

def worker(stream):
    print("Start: %s" % (stream))
    methods = config.methods()
    ttt = TestThenTrain(metrics=list(config.metrics().values()), verbose=False)
    ttt.process(streams[stream], methods)
    np.save("results/ex2_rev_all/%s" % stream, ttt.scores)
    print("End: %s" % (stream))

jobs = []
for replication in range(replications):
    streams = config.streams(config.random_state() + replication)
    for stream in streams:
        p = multiprocessing.Process(target=worker, args=(stream,))
        jobs.append(p)
        p.start()
