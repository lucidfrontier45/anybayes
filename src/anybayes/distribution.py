from abc import ABC, abstractmethod
from typing import Self

from nptyping import Float, NDArray, Number, Shape


class Distribution(ABC):
    @abstractmethod
    def fit(self, X: NDArray[Shape["N, D"], Number]) -> Self:
        pass

    @abstractmethod
    def pdf(self, X: NDArray[Shape["N, D"], Number]) -> NDArray[Shape["N, D"], Float]:
        pass
