from dataclasses import dataclass
from typing import Any, Literal, Self, TypeAlias

import numpy as np
from nptyping import Float, NDArray, Shape
from sklearn.neighbors import KernelDensity

from .base import KDEBackend

BandwidthType: TypeAlias = float | Literal["scott", "silverman"]
KernelType: TypeAlias = Literal[
    "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
]
MetricType: TypeAlias = str
MetricParamsType: TypeAlias = dict[str, Any] | None


@dataclass
class SKLearnKDEBackend(KDEBackend):
    bandwidth: BandwidthType = "scott"
    kernel: KernelType = "gaussian"
    metric: MetricType = "euclidean"
    metric_params: MetricParamsType = None

    def fit(self, X: NDArray[Shape["N, D"], Float]) -> Self:
        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            metric=self.metric,
            metric_params=self.metric_params,
        ).fit(X)
        return self

    def evaluate(
        self, X: NDArray[Shape["N, D"], Float]
    ) -> NDArray[Shape["N, D"], Float]:
        return np.exp(self.kde_.score_samples(X))


@dataclass
class IndependentSKLearnKDEBackend(KDEBackend):
    bandwidth: BandwidthType | list[BandwidthType] = "scott"
    kernel: KernelType | list[KernelType] = "gaussian"
    metric: MetricType | list[MetricType] = "euclidean"
    metric_params: MetricParamsType | list[MetricParamsType] = None

    def fit(self, X: NDArray[Shape["N, D"], Float]) -> Self:
        n_features = X.shape[1]

        bandwidth_list: list[BandwidthType]
        if isinstance(self.bandwidth, list):
            if len(self.bandwidth) != n_features:
                raise ValueError(
                    f"Expected {n_features} bandwidths, got {len(self.bandwidth)}"
                )
            bandwidth_list = self.bandwidth
        else:
            bandwidth_list = [self.bandwidth] * n_features

        kernel_list: list[KernelType]
        if isinstance(self.kernel, list):
            if len(self.kernel) != n_features:
                raise ValueError(
                    f"Expected {n_features} kernels, got {len(self.kernel)}"
                )
            kernel_list = self.kernel
        else:
            kernel_list = [self.kernel] * n_features

        metric_list: list[MetricType]
        if isinstance(self.metric, list):
            if len(self.metric) != n_features:
                raise ValueError(
                    f"Expected {n_features} metrics, got {len(self.metric)}"
                )
            metric_list = self.metric
        else:
            metric_list = [self.metric] * n_features

        metric_params_list: list[MetricParamsType]
        if isinstance(self.metric_params, list):
            if len(self.metric_params) != n_features:
                raise ValueError(
                    f"Expected {n_features} metric_params, got {len(self.metric_params)}"
                )
            metric_params_list = self.metric_params
        else:
            metric_params_list = [self.metric_params] * n_features

        self.kdes_ = []
        for i in range(n_features):
            self.kdes_.append(
                SKLearnKDEBackend(
                    bandwidth=bandwidth_list[i],
                    kernel=kernel_list[i],
                    metric=metric_list[i],
                    metric_params=metric_params_list[i],
                ).fit(X[:, i : i + 1])
            )
        return self

    def evaluate(
        self, X: NDArray[Shape["N, D"], Float]
    ) -> NDArray[Shape["N, D"], Float]:
        probs = []
        for i, kde in enumerate(self.kdes_):
            probs.append(kde.evaluate(X[:, i : i + 1]))
        return np.prod(probs, axis=0).T.copy()
