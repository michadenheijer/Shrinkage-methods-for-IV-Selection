import numpy as np
from sklearn.linear_model import ElasticNet


class LassoIC:
    """
    Select alpha and l1_ratio for ElasticNet using an Information Criterion (AIC or BIC)
    computed on the entire dataset.
    """

    def __init__(
        self,
        alphas=None,
        criterion="bic",
        random_state=None,
        max_iter=1000,
        fit_intercept=True,
        tol=1e-4,
    ):
        """
        Parameters
        ----------
        alphas : array-like, optional
            List of alpha values to try. If None, uses a default log-spaced range.
        criterion : {'aic', 'bic'}
            Which information criterion to use for selection.
        random_state : int or None
            Seed for the random number generator (for solver reproducibility).
        max_iter : int
            Maximum number of iterations for the ElasticNet solver.
        fit_intercept : bool
            Whether to fit an intercept in the model.
        tol : float
            Tolerance for the optimization.
        """
        if alphas is None:
            alphas = np.logspace(-4, 1, 10)  # e.g. 1e-4 ... 10

        self.alphas = alphas

        self.criterion = criterion.lower()
        if self.criterion not in ("aic", "bic"):
            raise ValueError("criterion must be either 'aic' or 'bic'")

        self.random_state = random_state
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.tol = tol

        # Best model info after fit
        self.best_alpha_ = None
        self.best_ic_ = np.inf
        self.best_model_ = None

        # Store search results
        self.search_results_ = []

    def fit(self, X, y):
        """
        Fit an ElasticNet model for each (alpha) in the grid on the entire dataset,
        compute the chosen IC, and pick the best.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        best_ic = np.inf
        best_model = None
        best_alpha = None

        for alpha in self.alphas:
            # Fit ElasticNet on entire dataset
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=1,
                max_iter=self.max_iter,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
                tol=self.tol,
            )
            model.fit(X, y)

            # Compute residual sum of squares (RSS)
            y_pred = model.predict(X)
            rss = np.sum((y - y_pred) ** 2)

            # Approx. degrees of freedom = # nonzero coefs + 1 (if intercept is used)
            # (This is a common heuristic for L1-based methods.)
            df = np.sum(np.abs(model.coef_) > 1e-15)
            if self.fit_intercept:
                df += 1

            # Estimate of variance: sigma^2 ~ RSS / n
            sigma2 = rss / n_samples if n_samples > 0 else 1.0

            # AIC/BIC formulas (dropping constant terms)
            if self.criterion == "bic":
                # BIC = n * ln(sigma^2) + df * ln(n)
                ic_value = n_samples * np.log(sigma2) + df * np.log(n_samples)
            else:  # AIC
                # AIC = n * ln(sigma^2) + 2 * df
                ic_value = n_samples * np.log(sigma2) + 2 * df

            # Save result
            self.search_results_.append(
                {
                    "alpha": alpha,
                    "df": df,
                    "rss": rss,
                    self.criterion: ic_value,
                }
            )

            # Track best
            if ic_value < best_ic:
                best_ic = ic_value
                best_model = model
                best_alpha = alpha

        # Store final best
        self.best_ic_ = best_ic
        self.best_model_ = best_model
        self.best_alpha_ = best_alpha

        return self.best_model_

    def predict(self, X):
        """
        Predict with the best model found during IC-based selection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if self.best_model_ is None:
            raise ValueError("No model has been fitted yet. Call fit() first.")
        return self.best_model_.predict(X)

    def get_ic_results(self):
        """
        Retrieve the full grid of (alpha, df, rss, IC) results.

        Returns
        -------
        results : list of dict
            Each dict has keys ['alpha', 'df', 'rss', <IC name>].
        """
        return self.search_results_
