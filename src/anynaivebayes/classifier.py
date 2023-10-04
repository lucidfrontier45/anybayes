from typing import Callable

import numpy as np
from nptyping import Float, Int, NDArray, Number, Shape
from sklearn.base import BaseEstimator, ClassifierMixin

from .distribution import Distribution


class AnyNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, distribution_factory: Callable[[], Distribution]):
        self.distribution_factory = distribution_factory
        self.dists_: list[Distribution] = []
        self.n_classes_: int = 0

    def fit(self, X: NDArray[Shape["N, D"], Number], y: NDArray[Shape["N"], Int]):
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
        probs = self.predict_proba(X, class_weight=class_weight)
        return np.argmax(probs, axis=1)
