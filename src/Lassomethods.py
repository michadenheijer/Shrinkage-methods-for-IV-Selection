from sklearn.linear_model import LassoCV, Lasso
from scipy.stats import norm
import numpy as np


class LassoVariant:
    """
    A class to handle different Lasso variants and their settings.
    """

    def __init__(
        self, n_instruments, n_samples, method="lasso_cv", seed=None, **kwargs
    ):
        self.method = method.lower()
        self.seed = seed
        self.kwargs = kwargs
        self.model = self.model_selection()
        self.n_instruments = n_instruments
        self.n_samples = n_samples

    def model_selection(self):
        """Returns the model selection method."""
        if self.method == "standard_lasso":
            if self.kwargs["lambda_method"] == "cv":
                return LassoCV(
                    cv=self.kwargs["cv"],
                    random_state=self.seed,
                    max_iter=self.kwargs["max_iter"],
                )
            elif self.kwargs["lambda_method"] == "Xdependent":
                return Lasso(max_iter=self.kwargs["max_iter"])
            elif self.kwargs["lambda_method"] == "Xindependent":
                return Lasso(max_iter=self.kwargs["max_iter"])
        else:
            raise ValueError(f"Unknown Lasso method: {self.method}")

    def fit(self, X, y):
        """Fits the Lasso model to the data."""
        if (
            self.method == "standard_lasso"
            and self.kwargs["lambda_method"] == "Xindependent"
        ):
            alpha = (
                2
                * self.kwargs["c"]
                * np.sqrt(np.var(X))
                * np.sqrt(self.n_samples)
                * norm.ppf(1 - self.kwargs["alpha"] / (2 * self.n_instruments))
            ) / self.n_samples  # NOTE this division by n_samples is missing in the original paper, but that is due to different calculation by the package
            self.model.set_params(alpha=alpha)

        if (
            self.method == "standard_lasso"
            and self.kwargs["lambda_method"] == "Xdependent"
        ):
            alpha = (
                2
                * self.kwargs["c"]
                * np.sqrt(np.var(X))
                * norm.ppf(1 - self.kwargs["alpha"] / (2 * self.n_samples))
            )
            self.model.set_params(alpha=alpha)

        self.model = self.model.fit(X, y)

    def selected_features(self):
        """Returns indices of the selected features."""
        return np.where(self.model.coef_ != 0)[0]

    def coefficients(self):
        """Returns model coefficients."""
        return self.model.coef_
