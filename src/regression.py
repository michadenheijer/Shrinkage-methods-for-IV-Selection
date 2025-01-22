import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from linearmodels.iv import IVLIML, IV2SLS


class RegressionModel:
    """
    A class to handle different regression methods.
    """

    def __init__(self, method):
        self.method = method.lower()
        self.model = None

    def fit(self, dependent, exog, endog, instruments):
        """Fits the regression model to the data."""
        if self.method == "2sls":
            self.model = IV2SLS(dependent, exog, endog, instruments).fit()
        elif self.method == "fuller":
            self.model = IVLIML(dependent, exog, endog, instruments, fuller=1).fit()
        else:
            raise ValueError(f"Unknown regression method: {self.method}")

    def predict(self, X):
        """Predicts using the fitted model."""
        return self.model.predict(X)

    def coefficients(self):
        """Returns model coefficients."""
        return self.model.params

    def p_values(self):
        """Returns p-values of the coefficients"""
        return self.model.pvalues
