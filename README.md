# AnyBayes
A Bayesian Classifier with Any Distribution

# Install

Please first install PDM >= 2.0 with pip/pipx.

```bash
pdm install --prod
```

# Develop

```bash
pdm install
```

# VSCode Settings

```bash
cp vscode_templates .vscode
```

Then install/activate all extensions listed in `.vscode/extensions.json`

# Usage

## API

```py
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
        Fit the Naive Bayes classifier to the training data.
    predict_proba(X, class_weight=1.0)
        Predict the class probabilities for the given test data.
    predict(X, class_weight=1.0)
        Predict the class labels for the given test data.
    """

    def __init__(self, distribution_factory: Callable[[], Distribution]):
        ...

    def fit(
        self, X: NDArray[Shape["N, D"], Number], y: NDArray[Shape["N"], Int]
    ) -> Self:
        """
        Fit the Naive Bayes classifier to the training data.

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
        ...

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
        ...

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
        ...
```

## Example

Check notebooks in the `examples` directory.

## Implement Custom Backend Distribution

This package currently only includes empirical distribution backed by scikit-learn's KDE. If you want to use other distributions you need to add custom wrapper class that implements `Distribution` abstract class. For more detail, please check `src/anybayes/backends/kde.py` to understand how it is implemented. 