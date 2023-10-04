from typing import Callable

import numpy as np
from nptyping import Float, Int, NDArray, Shape
from sklearn.base import BaseEstimator, ClassifierMixin

from .backends.base import KDEBackend
from .version import __version__


class KDENaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, backend_factory: Callable[[], KDEBackend]):
        self.backend_factory = backend_factory
        self.kdes_: list[KDEBackend] = []
        self.n_classes_: int = 0

    def fit(self, X: NDArray[Shape["N, D"], Float], y: NDArray[Shape["N"], Int]):
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
    ) -> NDArray[Shape["N, D"], Float]:
        if self.n_classes_ == 0:
            raise RuntimeError("You must fit the model before predicting")

        if isinstance(class_weight, float):
            class_weights = [class_weight] * self.n_classes_
        else:
            if len(class_weight) != self.n_classes_:
                raise ValueError(
                    f"Expected {self.n_classes_} class weights, got {len(class_weight)}"  # noqa: E501
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
        probs = self.predict_proba(X, class_weight=class_weight)
        return np.argmax(probs, axis=1)
