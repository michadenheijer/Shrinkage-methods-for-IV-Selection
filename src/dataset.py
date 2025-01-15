import numpy as np

def simulate_dataset(config, seed=None):
    """Generates the data based on the configuration settings."""
    n_samples = config["dgp"]["n_samples"]
    n_instruments = config["dgp"]["n_instruments"] # Is set to 100 in Spindler
    mu2 = config["dgp"]["mu2"] 
    beta_true = config["dgp"]["beta_true"] # Is set to 1 in Spindler
    sigma_e = config["dgp"]["sigma_e"] # Is set to 1 in Spindler
    rho_ev = config["dgp"]["rho_ev"] # Is set to 0.6 in Spindler
    sigma_z = config["dgp"]["sigma_z"] # Is set to 1 in Spindler
    correlation = config["dgp"]["correlation"] # Is set to 0.5 in Spindler
    
    # Set seed if provided
    if not seed is None:
        np.random.seed(seed)
    
    # Generate instruments (Z)
    cov_matrix = sigma_z * (correlation ** np.abs(np.subtract.outer(range(n_instruments), range(n_instruments))))
    Z = np.random.multivariate_normal(np.zeros(n_instruments), cov_matrix, size=n_samples)

    # Calculate the Pi tilde matrix
    if config["dgp"]["design"] == "Exponential":
        Pi_tilde = np.array([0.7**i for i in range(n_instruments)])
    elif config["dgp"]["design"] in [5, 50]:
        Pi_tilde = np.concatenate((np.ones(config["dgp"]["design"]), np.zeros(n_instruments - config["dgp"]["design"])))
    else:
        raise ValueError(f"Unknown design: {config['dgp']['design']}")
    
    # Calculate C
    denominator = n_samples * Pi_tilde.T @ cov_matrix @ Pi_tilde + mu2 * Pi_tilde.T @ cov_matrix @ Pi_tilde
    C = np.sqrt(mu2 / denominator)
    
    # Calculate the Pi matrix
    Pi = C * Pi_tilde

    # Generate error terms (v, e)
    sigma_v = 1 - Pi.T @ cov_matrix @ Pi
    cov_matrix_error = np.array([[sigma_e, rho_ev], [rho_ev, sigma_v]])
    errors = np.random.multivariate_normal([0, 0], cov_matrix_error, size=n_samples)
    
    # Generate endogenous regressor (d)
    d = Z @ Pi + errors[:, 1]
    
    # Generate outcome variable (y)
    y = beta_true * d + errors[:, 0]
    
    data = {"Z": Z, "d": d, "y": y}

    return data