from sklearn.linear_model import LassoCV
import numpy as np

class LassoVariant:
    """
    A class to handle different Lasso variants and their settings.
    """
    def __init__(self, method="lasso_cv", **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        """Fits the Lasso model to the data."""
        if self.method == "lasso_cv":
            self.model = LassoCV(cv=10, random_state=42, **self.kwargs).fit(X, y)
        else:
            raise ValueError(f"Unknown Lasso variant: {self.method}")

    def selected_features(self):
        """Returns indices of the selected features."""
        return np.where(self.model.coef_ != 0)[0]

    def coefficients(self):
        """Returns model coefficients."""
        return self.model.coef_