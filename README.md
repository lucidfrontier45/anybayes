# KDE Naive Bayes
Naive Bayes Classifier with Empirical Distribution by Kernel Density Estimation

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
        ...

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
        ...

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
        ...
```

## Example

Check notebooks in the `examples` directory.

## Using Custom KDE Backend

This package currently only includes scikit-learn's KDE implementaton. If you want to use others (e.g. scipy, statsmodels), you need to add custom wrapper class that implements `KDEBackend` abstract class. For more detail, please check `src/kdenaivebayes/backends/sklearn.py` to understand how it is implemented. 