import numpy as np
from sklearn.model_selection import GridSearchCV
from skglm import MCPRegression


class MCPRegressionCV:
    def __init__(self, alphas=None, gammas=None, cv=5, random_state=42, max_iter=1000):
        """
        Custom cross-validation for MCPRegression.

        Parameters:
        - alphas: List or array of alpha values to try.
        - gammas: List or array of gamma values to try.
        - cv: Number of folds for cross-validation.
        """
        # TODO: Alpha and gamma values are just randomly suggested by chat
        if alphas is None:
            alphas = np.logspace(
                -4, 1, 10
            )  # NOTE may need to increase values in between
        if gammas is None:
            gammas = np.linspace(1.5, 4, 6)  # NOTE same here
        self.alphas = alphas
        self.gammas = gammas
        self.cv = cv
        self.best_pair = None
        self.best_model = None
        self.grid_search = None
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the MCPRegression model with cross-validation.

        Parameters:
        - X: Input features, array-like, shape (n_samples, n_features).
        - y: Target values, array-like, shape (n_samples,).
        """
        X = np.ascontiguousarray(
            X, dtype=np.float64
        )  # Maakt het op een of andere manier sneller
        y = np.ascontiguousarray(y, dtype=np.float64)
        param_grid = {"alpha": self.alphas, "gamma": self.gammas}
        self.grid_search = GridSearchCV(
            MCPRegression(max_iter=self.max_iter, warm_start=True),
            param_grid,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,  # FIXME weet niet of dit logisch is als ergens anders al parallelisatie is
        )

        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_

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
        """
        return self.grid_search.cv_results_


class MCPRegressionIC:
    def __init__(
        self, alphas=None, gammas=None, criterion="bic", random_state=42, max_iter=1000
    ):
        """
        Perform MCPRegression hyperparameter selection using an Information Criterion (AIC or BIC).

        Parameters
        ----------
        alphas : array-like
            List or array of alpha values to try (regularization strength).
        gammas : array-like
            List or array of gamma values to try (MCP parameter).
        criterion : {'aic', 'bic'}
            Which information criterion to use for selection.
        random_state : int
            Random seed for reproducibility in the MCPRegression solver.
        max_iter : int
            Maximum number of iterations for MCPRegression solver.
        """
        if alphas is None:
            # Example grid of alpha values (log-spaced)
            alphas = np.logspace(-4, 1, 10)
        if gammas is None:
            # Example grid of gamma values
            gammas = np.linspace(1.5, 4, 6)

        self.alphas = alphas
        self.gammas = gammas
        self.criterion = criterion.lower()
        if self.criterion not in ["aic", "bic"]:
            raise ValueError("criterion must be either 'aic' or 'bic'")

        self.random_state = random_state
        self.max_iter = max_iter

        self.best_alpha_ = None
        self.best_gamma_ = None
        self.best_ic_ = np.inf
        self.best_model_ = None

        # Will store search results
        self.search_results_ = []

    def fit(self, X, y):
        """
        Fit MCPRegression models for each (alpha, gamma) in the grid on the entire dataset,
        compute the chosen Information Criterion (IC), and pick the best combination.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n_samples = X.shape[0]

        best_ic = np.inf
        best_model = None
        best_alpha = None
        best_gamma = None

        # Exhaustive search over alphas and gammas
        for alpha in self.alphas:
            for gamma in self.gammas:
                model = MCPRegression(
                    alpha=alpha,
                    gamma=gamma,
                    max_iter=self.max_iter,
                    warm_start=True,
                )
                model.fit(X, y)

                # Compute RSS on full data
                y_pred = model.predict(X)
                rss = np.sum((y - y_pred) ** 2)

                # Approx. degrees of freedom = #nonzero coefs + 1 (if intercept)
                df = np.sum(np.abs(model.coef_) > 1e-15)
                if getattr(model, "fit_intercept", True):
                    df += 1

                # Residual variance estimate: sigma^2 ~ RSS / n
                # (Alternatively, RSS/(n - df) could be used for an "unbiased" estimate.)
                # We'll just do RSS / n for simplicity.
                sigma2 = (
                    rss / n_samples if n_samples > 0 else 1.0
                )  # avoid zero division

                if self.criterion == "bic":
                    # BIC = n log(sigma^2) + df log(n)
                    ic_value = n_samples * np.log(sigma2) + df * np.log(n_samples)
                else:  # AIC
                    # AIC = n log(sigma^2) + 2 df
                    ic_value = n_samples * np.log(sigma2) + 2 * df

                self.search_results_.append(
                    {
                        "alpha": alpha,
                        "gamma": gamma,
                        self.criterion: ic_value,
                        "df": df,
                        "rss": rss,
                    }
                )

                # Track best (lowest) IC
                if ic_value < best_ic:
                    best_ic = ic_value
                    best_model = model
                    best_alpha = alpha
                    best_gamma = gamma

        self.best_alpha_ = best_alpha
        self.best_gamma_ = best_gamma
        self.best_ic_ = best_ic
        self.best_model_ = best_model

        return self.best_model_

    def predict(self, X):
        """
        Predict using the best model found during IC-based selection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.best_model_ is None:
            raise ValueError("No model has been fitted yet. Call fit() first.")
        return self.best_model_.predict(X)

    def get_ic_results(self):
        """
        Return the list of dicts with all (alpha, gamma, IC) evaluations.

        Returns
        -------
        results : list of dict
            Each dict contains {'alpha', 'gamma', <IC-name>, 'df', 'rss'}.
        """
        return self.search_results_
