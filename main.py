# In[]:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.regression import RegressionModel
from src.Lassomethods import LassoVariant
import yaml

# In[]:
CONFIG_PATH = "configs/configJasper.yaml"

def simulate_dataset(config):
    """Generates the data based on the configuration settings."""
    n_samples = config["dgp"]["n_samples"]
    n_instruments = config["dgp"]["n_instruments"]
    beta_true = config["dgp"]["beta_true"]
    sigma_e = config["dgp"]["sigma_e"]
    sigma_v = config["dgp"]["sigma_v"]
    sigma_z = config["dgp"]["sigma_z"]
    correlation = config["dgp"]["correlation"]

    # Generate instruments (Z)
    cov_matrix = sigma_z * (correlation ** np.abs(np.subtract.outer(range(n_instruments), range(n_instruments))))
    Z = np.random.multivariate_normal(np.zeros(n_instruments), cov_matrix, size=n_samples)

    # Generate endogenous regressor (d)
    gamma = np.linspace(1, 0.7, n_instruments)
    v = np.random.normal(0, sigma_v, size=n_samples)
    d = Z @ gamma + v

    # Generate outcome variable (y)
    e = np.random.normal(0, sigma_e, size=n_samples)
    y = beta_true * d + e
    
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

    # Generate data
    data = simulate_dataset(config)

    # In[]: Stage 1: Lasso for variable selection
    lasso = LassoVariant(method=config["lasso"]["method"], **config["lasso"]["kwargs"])
    lasso.fit(data["Z"], data["d"])
    selected_features = lasso.selected_features()

    # Use selected features
    Z_selected = data["Z"][:, selected_features]

    # In[]: Stage 2: Regression
    reg_model = RegressionModel(method=config["regression"]["method"])
    reg_model.fit(data["Z"], data["y"])

    # Evaluate the model
    y_pred = reg_model.predict(data["Z"])
    mse = mean_squared_error(data["y"], y_pred)

    # Results
    print(f"Selected instruments: {selected_features}")
    print(f"Post-Lasso coefficients: {reg_model.coefficients()}")
    print(f"Mean Squared Error: {mse}")
