from sklearn.linear_model import LassoCV, Lasso
import numpy as np

class LassoVariant:
    """
    A class to handle different Lasso variants and their settings.
    """
    def __init__(self, method="lasso_cv", seed=None, **kwargs):
        self.method = method.lower()
        self.seed = seed
        self.kwargs = kwargs
        self.model = self.model_selection()

    def model_selection(self):
        """Returns the model selection method."""
        if self.method == "standard_lasso":
            if self.kwargs['lambda_method'] == "cv":
                return LassoCV(cv=self.kwargs['cv'], random_state=self.seed, max_iter=self.kwargs['max_iter'])
            elif self.kwargs['lambda_method'] == "Xdependent":
                raise NotImplementedError("Xdependent lambda not implemented")
            elif self.kwargs['lambda_method'] == "Xindependent":
                raise NotImplementedError("Xindependent lambda not implemented")
        else:
            raise ValueError(f"Unknown Lasso method: {self.method}")
            
    def fit(self, X, y):
        """Fits the Lasso model to the data."""
        self.model = self.model.fit(X, y)

    def selected_features(self):
        """Returns indices of the selected features."""
        return np.where(self.model.coef_ != 0)[0]

    def coefficients(self):
        """Returns model coefficients."""
        return self.model.coef_