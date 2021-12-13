"""Online Active Learning Ensemble."""

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import matplotlib.pyplot as plt


class OALE(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator=None, n_estimators=10, metric=accuracy_score):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.metric = metric

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.counter_ = 0
            self.weights = np.zeros((1, self.n_estimators)).astype(int)

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Append new estimator
        self.ensemble_.append(clone(self.base_estimator).fit(self.X_, self.y_))

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            del self.ensemble_[
                np.argmin([self.metric(y, clf.predict(X)) for clf in self.ensemble_])
            ]

        # PLOT
        # ####################################################################
        if self.counter_ < self.n_estimators:
            self.weights[0,self.counter_] = self.counter_
        else:
            del_ind = np.argmin([self.metric(y, clf.predict(X)) for clf in self.ensemble_])
            self.weights = np.delete(self.weights, del_ind, 1)
            self.weights = np.append(self.weights, [[self.counter_]]).reshape(1,-1)

        clfs_names = ["CLF %i" % (i+1) for i in range(len(self.ensemble_))]

        fig, ax = plt.subplots()
        im = ax.imshow(self.weights)

        ax.set_xticks(np.arange(len(clfs_names)))
        ax.set_xticklabels(clfs_names)
        ax.set_yticklabels([""])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for j in range(len(clfs_names)):
            text = ax.text(j, 0, self.weights[0, j],
                           ha="center", va="center", color="w")

        ax.set_title("Ensemble")
        fig.tight_layout()
        plt.savefig("ensemble.png", dpi=120)
        plt.close()
        # ####################################################################

        self.counter_ += 1
        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
