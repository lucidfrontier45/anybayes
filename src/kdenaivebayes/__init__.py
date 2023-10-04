from typing import Callable

import numpy as np
from nptyping import Float, Int, NDArray, Shape
from sklearn.base import BaseEstimator, ClassifierMixin

from .backends.base import KDEBackend
from .version import __version__


class KDENaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    A Naive Bayes classifier using kernel density estimation (KDE) to estimate the class-conditional densities.

    This class assumes that the features are independent given the class label, hence the "naive" assumption.

    Parameters
    ----------
    backend_factory : Callable[[], KDEBackend]
        A callable that returns an instance of a `KDEBackend` class. This is used to create the KDE estimators for each class.

    Attributes
    ----------
    backend_factory : Callable[[], KDEBackend]
        The callable that returns an instance of a `KDEBackend` class.
    kdes_ : list[KDEBackend]
        A list of KDE estimators, one for each class.
    n_classes_ : int
        The number of classes in the training data.

    Methods
    -------
    fit(X, y)
        Fit the Naive Bayes classifier to the training data.
    predict_proba(X, class_weight=1.0)
        Predict the class probabilities for the given test data.
    predict(X, class_weight=1.0)
        Predict the class labels for the given test data.
    """

    def __init__(self, backend_factory: Callable[[], KDEBackend]):
        self.backend_factory = backend_factory
        self.kdes_: list[KDEBackend] = []
        self.n_classes_: int = 0

    def fit(self, X: NDArray[Shape["N, D"], Float], y: NDArray[Shape["N"], Int]):
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            The training data, where N is the number of samples and D is the number of features.
        y : numpy.ndarray, shape (N,)
            The class labels for each sample.

        Returns
        -------
        self : KDENaiveBayesClassifier
            The fitted KDENaiveBayesClassifier instance.
        """
        # assumes y is a list of integers which starts from 0 without gaps
        self.n_classes_ = max(y) + 1
        self.kdes_ = []
        for class_ in range(self.n_classes_):
            mask = y == class_
            x = X[mask]
            kde = self.backend_factory().fit(x)
            self.kdes_.append(kde)
        return self

    def predict_proba(
        self,
        X: NDArray[Shape["N, D"], Float],
        class_weight: list[float] | float = 1.0,
    ) -> NDArray[Shape["N, C"], Float]:
        """
        Predict the class probabilities for the given test data.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            The test data, where N is the number of samples and D is the number of features.
        class_weight : float or list[float], optional
            The weight of each class. If a float is given, it is used as the weight for all classes.
            If a list is given, it must have the same length as the number of classes.

        Returns
        -------
        probs : numpy.ndarray, shape (N, C)
            The predicted class probabilities for each sample, where C is the number of classes.
        """
        if self.n_classes_ == 0:
            raise RuntimeError("You must fit the model before predicting")

        if isinstance(class_weight, float):
            class_weights = [class_weight] * self.n_classes_
        else:
            if len(class_weight) != self.n_classes_:
                raise ValueError(
                    f"Expected {self.n_classes_} class weights, got {len(class_weight)}"
                )
            class_weights = class_weight
        probs = []
        for w, kde in zip(class_weights, self.kdes_):
            probs.append(kde.evaluate(X) * w)
        p = np.asarray(probs, dtype=np.float64).T.copy()
        return p / p.sum(axis=1, keepdims=True)

    def predict(
        self,
        X: NDArray[Shape["N, D"], Float],
        class_weight: list[float] | float = 1.0,
    ) -> NDArray[Shape["N"], Int]:
        """
        Predict the class labels for the given test data.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            The test data, where N is the number of samples and D is the number of features.
        class_weight : float or list[float], optional
            The weight of each class. If a float is given, it is used as the weight for all classes.
            If a list is given, it must have the same length as the number of classes.

        Returns
        -------
        labels : numpy.ndarray, shape (N,)
            The predicted class labels for each sample.
        """
        probs = self.predict_proba(X, class_weight=class_weight)
        return np.argmax(probs, axis=1)
