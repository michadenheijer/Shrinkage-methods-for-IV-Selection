import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from statsmodels.api import OLS

class RegressionModel:
    """
    A class to handle different regression methods.
    """
    def __init__(self, method="ols"):
        self.method = method.lower()
        self.model = None

    def fit(self, X, y):
        """Fits the regression model to the data."""
        if self.method == "ols":
            self.model = OLS(y, X).fit()
        elif self.method == "2sls":
            # Implement 2SLS here
            #TODO: Jasper implementeren
            pass
        elif self.method == 'fuller':
            # Implement Fuller here
            #TODO: Micha implementern
            pass
        else:
            raise ValueError(f"Unknown regression method: {self.method}")

    def predict(self, X):
        """Predicts using the fitted model."""
        if self.method == "ols":
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def coefficients(self):
        """Returns model coefficients."""
        if self.method == "ols":
            return self.model.params
        else:
            return self.model.coef_