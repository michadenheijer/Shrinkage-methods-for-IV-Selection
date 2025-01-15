# In[]:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.regression import RegressionModel
from src.Lassomethods import LassoVariant
import yaml

# In[]:
CONFIG_PATH = "configs/configJasper.yaml"

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

def load_config(config_path):
    """Loads settings from a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
# In[]:

# Example usage
if __name__ == "__main__":
    # In[]: # Load configuration
    config = load_config(CONFIG_PATH)
    seed = None if config["seed"] == "None" else config["seed"]

    # Generate data
    data = simulate_dataset(config, seed)

    # In[]: Stage 1: Lasso for variable selection
    lasso = LassoVariant(method=config["lasso"]["method"], seed=seed, **config["lasso"]["kwargs"])
    lasso.fit(data["Z"], data["d"])
    selected_features = lasso.selected_features()

    # Use selected features
    Z_selected = data["Z"][:, selected_features]
    print(f"Number of selected instruments: {Z_selected.shape[1]}")

    # In[]: Stage 2: Regression
    constant = np.ones((len(Z_selected), 1))  # Add constant term (required for 2SLS)
    reg_model = RegressionModel(method=config["regression"]["method"])
    reg_model.fit(dependent=data["y"], exog=constant, endog=data["d"], instruments=Z_selected)

    # Evaluate the model
    # TODO: The evaluation doen
    # y_pred = reg_model.predict(data["Z"])
    # mse = mean_squared_error(data["y"], y_pred)

    # # Results
    # print(f"Selected instruments: {selected_features}")
    # print(f"Post-Lasso coefficients: {reg_model.coefficients()}")
    # print(f"Mean Squared Error: {mse}")
