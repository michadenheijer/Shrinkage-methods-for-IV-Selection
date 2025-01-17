import numpy as np
from sklearn.model_selection import KFold
from skglm import MCPRegression
from sklearn.metrics import mean_squared_error


class MCPRegressionCV:
    def __init__(self, alphas=None, cv=5, random_state=42, max_iter=1000):
        """
        Custom cross-validation for MCPRegression.

        Parameters:
        - alphas: List or array of alpha values to try.
        - cv: Number of folds for cross-validation.
        """
        # TODO: Alpha values are just randomly suggested by chat
        # TODO: Gamma not yet tested, could be very slow
        if alphas is None:
            alphas = np.logspace(-4, 1, 50)
        self.alphas = alphas
        self.cv = cv
        self.best_alpha = None
        self.best_model = None
        self.cv_errors = []
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the MCPRegression model with cross-validation.

        Parameters:
        - X: Input features, array-like, shape (n_samples, n_features).
        - y: Target values, array-like, shape (n_samples,).
        """
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        mean_errors = []

        for alpha in self.alphas:
            fold_errors = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = MCPRegression(alpha=alpha, max_epochs=self.max_iter)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                fold_errors.append(mean_squared_error(y_val, y_pred))

            mean_errors.append(np.mean(fold_errors))

        self.cv_errors = mean_errors
        self.best_alpha = self.alphas[np.argmin(mean_errors)]
        self.best_model = MCPRegression(alpha=self.best_alpha).fit(X, y)

        return self.best_model

    def predict(self, X):
        """
        Predict using the best model found during cross-validation.

        Parameters:
        - X: Input features, array-like, shape (n_samples, n_features).

        Returns:
        - Predictions: array-like, shape (n_samples,).
        """
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet. Call `fit` first.")
        return self.best_model.predict(X)

    def get_cv_results(self):
        """
        Get cross-validation results.

        Returns:
        - alphas: List of alpha values used in cross-validation.
        - errors: Corresponding mean errors for each alpha.
        """
        return self.alphas, self.cv_errors
