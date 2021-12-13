"""Online Active Learning Ensemble."""

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import matplotlib.pyplot as plt


class OALE(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator=None, n_estimators=10, tiny=1,
                  norm_strategy='b'):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.tiny = tiny
        self.norm_strategy = norm_strategy

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
            self.weights = np.zeros(self.n_estimators+1)
            self.weights[0] = .5

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")

        # BLS
        # tmask = np.random.uniform(size=y.shape) <= self.tiny
        # self.X_, self.y_ = X[tmask], y[tmask]
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Append new estimators
        for i in range(2 if len(self.ensemble_) == 0 else 1):
            self.ensemble_.append(clone(self.base_estimator))

        # Train all this shit
        [clf.partial_fit(self.X_, self.y_, self.classes_)
         for clf in self.ensemble_]

        # Remove the oldest dynamic when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators+1:
                del self.ensemble_[1]

        if self.counter_ < self.n_estimators:
            self.weights[self.counter_+1] = 1/self.n_estimators
            self.weights[1:self.counter_+1] = self.weights[1:self.counter_+1] * (1-(1/self.n_estimators))

            # if self.counter_ > 1:
            if self.norm_strategy == 'a':
                self.norm = self.weights[1:] - np.min(self.weights[1:])
                self.norm = self.norm / (np.max(self.weights[1:])-np.min(self.weights[1:]))
                self.norm = np.append([[.5]], self.norm)
            elif self.norm_strategy == 'b':
                # Alternative
                self.norm = np.copy(self.weights)
                rrr = self.norm[1:]
                self.norm[1:] = rrr / (np.sum(rrr)*2)
            elif self.norm_strategy == 'c':
                # Alternative
                self.norm = np.linspace(.5,1,10)

        if self.norm_strategy == 'c':
            # Alternative
            self.norm = np.random.uniform(size=self.weights.shape)


        # PLOT
        """
        clfs_names = ["Base"] + ["CLF %i" % (i+1) for i in range(len(self.ensemble_)-1)]

        fig, ax = plt.subplots()
        im = ax.imshow([self.weights, self.norm], cmap='binary')

        ax.set_xticks(np.arange(len(clfs_names)))
        ax.set_xticklabels(clfs_names)
        ax.set_yticklabels([""])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                  rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for j in range(len(clfs_names)):
            val = self.weights[j]
            ax.text(j, 0, "%.3f" % val,
                    ha="center", va="center", color="black" if val < np.mean(self.weights) else 'white')

            val = self.norm[j]
            ax.text(j, 1, "%.3f" % val,
                    ha="center", va="center", color="black" if val < np.mean(self.norm) else 'white')

        ax.set_title("Ensemble %.3f - %.3f" % (np.sum(self.weights),
                                               np.sum(self.norm)))
        fig.tight_layout()
        plt.savefig("ensemble.png", dpi=120)
        plt.close()
        """

        self.counter_ += 1
        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) * self.norm[i]
                         for i, member_clf in enumerate(self.ensemble_)])

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
