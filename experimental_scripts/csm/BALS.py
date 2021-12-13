import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
import strlearn as sl
import sys


class BALS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, treshold=0.2, budget=0.05, random_state=None):
        self.treshold = treshold
        self.budget = budget
        self.base_estimator = base_estimator
        self.random_state = random_state

    def partial_fit(self, X, y, classes=None):
        np.random.seed(self.random_state)
        # First train
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            try:
                self.clf = clone(self.base_estimator).partial_fit(X, y, classes=classes)
            except:
                self.clf = self.base_estimator.partial_fit(X, y, classes=classes)
            self.usage = []

        else:
            supports = np.abs(self.clf.predict_proba(X)[:, 0] - 0.5)
            selected = supports < self.treshold

            # if np.sum(selected) > 0:
            #     self.clf.partial_fit(X[selected], y[selected], classes)
            #
            #     score = sl.metrics.balanced_accuracy_score(
            #         y[selected], self.clf.predict(X[selected])
            #     )
            #
            #     # self.treshold = 0.5 - score / 2
            #
            # self.usage.append(np.sum(selected) / selected.shape)

            # Get random subset
            limit = int(self.budget * len(y))
            idx = np.array(list(range(len(y))))
            selectedr = np.random.choice(idx, size=limit, replace=False)

            # Partial fit
            self.clf.partial_fit(np.concatenate((X[selected], X[selectedr]), axis=0), np.concatenate((y[selected], y[selectedr]), axis=0), classes)

    def predict(self, X):
        return self.clf.predict(X)
