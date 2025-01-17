from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, Lasso
from sklearn.model_selection import KFold
from scipy.stats import norm
import numpy as np

from .mcp import MCPRegressionCV


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

    def _model_selection_standard_lasso(self):
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
            raise ValueError(f"Unknown Lambda method: {self.method}")

    def _model_selection_elastic_net(self):
        if self.kwargs["lambda_method"] == "cv":
            return ElasticNetCV(
                cv=self.kwargs["cv"],
                random_state=self.seed,
                max_iter=self.kwargs["max_iter"],
                l1_ratio=self.kwargs["l1_ratio"],
            )
        else:
            raise ValueError(f"Unknown Lambda method: {self.method}")

    def _model_selection_minimax_concave_penalty(self):
        if self.kwargs["lambda_method"] == "cv":
            return MCPRegressionCV(
                cv=self.kwargs["cv"],
                random_state=self.seed,
                max_iter=self.kwargs["max_iter"],
            )
        else:
            raise ValueError(f"Unknown Lambda method: {self.method}")

    def model_selection(self):
        """Returns the model selection method."""
        if self.method == "standard_lasso":
            return self._model_selection_standard_lasso()
        elif self.method == "elastic_net":
            return self._model_selection_elastic_net()
        elif self.method == "minimax_concave_penalty":
            return self._model_selection_minimax_concave_penalty()
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

        # TODO fix this method as it is still incorrect
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
