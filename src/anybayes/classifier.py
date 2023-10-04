from typing import Callable, Self

import numpy as np
from nptyping import Float, Int, NDArray, Number, Shape
from sklearn.base import BaseEstimator, ClassifierMixin

from .distribution import Distribution


class AnyBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    A Bayesian classifier that can use any distribution for the class-conditional densities.

    Parameters
    ----------
    distribution_factory : Callable[[], Distribution]
        A callable that returns an instance of a `Distribution` class. This is used to create the density estimators
        for each class.

    Attributes
    ----------
    distribution_factory : Callable[[], Distribution]
        The callable that returns an instance of a `Distribution` class.
    dists_ : list[Distribution]
        A list of density estimators, one for each class.
    n_classes_ : int
        The number of classes in the training data.

    Methods
    -------
    fit(X, y)
        Fit the classifier to the training data.
    predict_proba(X, class_weight=1.0)
        Predict the class probabilities for the given test data.
    predict(X, class_weight=1.0)
        Predict the class labels for the given test data.
    """

    def __init__(self, distribution_factory: Callable[[], Distribution]):
        self.distribution_factory = distribution_factory
        self.dists_: list[Distribution] = []
        self.n_classes_: int = 0

    def fit(
        self, X: NDArray[Shape["N, D"], Number], y: NDArray[Shape["N"], Int]
    ) -> Self:
        """
        Fit the classifier to the training data.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            The training data, where N is the number of samples and D is the number of features.
        y : numpy.ndarray, shape (N,)
            The class labels for each sample. The labels are assumed to start from 0 without gaps

        Returns
        -------
        self : AnyBayesClassifier
            The fitted `AnyBayesClassifier` instance.
        """
        # assumes y is a list of integers which starts from 0 without gaps
        self.n_classes_ = max(y) + 1
        self.dists_ = []
        for class_ in range(self.n_classes_):
            mask = y == class_
            x = X[mask]
            dist = self.distribution_factory().fit(x)
            self.dists_.append(dist)
        return self

    def predict_proba(
        self,
        X: NDArray[Shape["N, D"], Number],
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
        for w, dist in zip(class_weights, self.dists_):
            probs.append(dist.pdf(X) * w)
        p = np.asarray(probs, dtype=np.float64).T.copy()
        return p / p.sum(axis=1, keepdims=True)

    def predict(
        self,
        X: NDArray[Shape["N, D"], Number],
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
