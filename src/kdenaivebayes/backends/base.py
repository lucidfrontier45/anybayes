from abc import ABC, abstractmethod
from typing import Self

from nptyping import Float, NDArray, Shape


class KDEBackend(ABC):
    @abstractmethod
    def fit(self, X: NDArray[Shape["N, D"], Float]) -> Self:
        pass

    @abstractmethod
    def evaluate(
        self, X: NDArray[Shape["N, D"], Float]
    ) -> NDArray[Shape["N, D"], Float]:
        pass
