import numpy as np
import strlearn as sl
import hashlib


def post_prunings():
    return (True, False)


def thetas():
    return (0, 0.025, 0.05, 0.075, 0.1)


def weight_calculation_methods():
    return ("same_for_each", "kuncheva", "pta_related_to_whole", "bell_curve")


def aging_methods():
    return ("weights_proportional", "constant", "gaussian")


def base_estimators():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from skmultiflow.trees import HoeffdingTreeClassifier

    return {
        "GNB": GaussianNB(),
        "HT": HoeffdingTreeClassifier(split_criterion='hellinger'),
        "MLP": MLPClassifier(random_state=1410),
    }


def methods():
    from strlearn.ensembles import WAE
    from csm import BALS

    _methods = []
    for base_estimator in base_estimators():
        for post_pruning in post_prunings():
            for theta in thetas():
                for wcm in weight_calculation_methods():
                    for aging_method in aging_methods():
                        clf = BALS(WAE(
                            base_estimators()[base_estimator],
                            theta=theta,
                            post_pruning=post_pruning,
                            weight_calculation_method=wcm,
                            aging_method=aging_method,
                        ), random_state=1410)
                        _methods.append(clf)
    return _methods


def replications():
    return 5


def random_state():
    return 1410


def properties_var():
    """
    Stream properties:
    1. n_chunks
    2. chunk_size
    """
    return (200, 250)


def metrics():
    from sklearn.metrics import accuracy_score

    return {"accuracy": accuracy_score}


def weights_var():
    """
    Class imbalance:
    - tuple for dynamic (n-drifts, sigmoid, amplitude)
    - list for static (positive should be minor)
    """
    return [[0.5, 0.5]]


def y_flip_var():
    """
    Label noise:
    - float for global label noise
    - tuple for separate label noise for every class
    """
    return [0.01]


def drift_var():
    """
    Concept drift:
    1. n_drifts
    2. concept_sigmoid_spacing (None for sudden)
    3. incremental [True] or gradual [False]
    4. recurring [True] or non-recurring [False]
    """
    return [(10, None, False, False), (10, 5, False, False), (10, 5, True, False)]


def concept_var():
    """
    Concept:
    1. n_features
    2. n_informative
    3. n_redundant
    4. n_repeated
    4. n_clusters_per_class
    """
    return [(8, 8, 0, 0, 1)]


def streams(random_state):
    """
    Build stream dictionary
    """
    streams = {}
    n_chunks, chunk_size = properties_var()
    for drift in drift_var():
        n_drifts, concept_sigmoid_spacing, incremental, recurring = drift
        for concept in concept_var():
            (
                n_features,
                n_informative,
                n_redundant,
                n_repeated,
                n_clusters_per_class,
            ) = concept
            for weights in weights_var():
                for y_flip in y_flip_var():
                    stream = sl.streams.StreamGenerator(
                        incremental=incremental,
                        weights=weights,
                        recurring=recurring,
                        random_state=random_state,
                        concept_sigmoid_spacing=concept_sigmoid_spacing,
                        y_flip=y_flip,
                        n_drifts=n_drifts,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_redundant=n_redundant,
                        n_repeated=n_repeated,
                        n_clusters_per_class=n_clusters_per_class,
                        n_chunks=n_chunks,
                        chunk_size=chunk_size,
                        class_sep=1,
                    )
                    streams.update({hash(stream_to_string(stream)): stream})
    return streams


def hash(string):
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def stream_to_string(stream):
    return "dr-%i-%s-%i-%i-co-%i-%i-%i-%i-%i-pr-%i-%i-we-%s-yf-%s-rs-%i" % (
        stream.n_drifts,
        "N"
        if stream.concept_sigmoid_spacing is None
        else "%i" % stream.concept_sigmoid_spacing,
        stream.incremental,
        stream.recurring,
        stream.n_features,
        stream.n_informative,
        stream.n_redundant,
        stream.n_repeated,
        stream.n_clusters_per_class,
        stream.n_chunks,
        stream.chunk_size,
        "D-%i-%i-%.2f" % stream.weights
        if type(stream.weights) is tuple
        else ("S-%s" % "-".join(["%.2f" % s for s in stream.weights])),
        "DYNDYN" if type(stream.y_flip) is tuple else "%.2f" % stream.y_flip,
        stream.random_state,
    )
