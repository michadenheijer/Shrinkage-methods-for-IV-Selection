import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.regression import RegressionModel
from src.Lassomethods import LassoVariant
import yaml

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

    return Z, d, y

def load_config(config_path):
    """Loads settings from a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Example usage
if __name__ == "__main__":
    # In[]: # Load configuration
    config = load_config(CONFIG_PATH)

    # Generate data
    Z, d, y = simulate_dataset(config)

    # Split data into training and testing sets
    Z_train, Z_test, d_train, d_test, y_train, y_test = train_test_split(Z, d, y, test_size=config["split"]["test_size"], random_state=config["split"]["random_state"])

    # In[]: Stage 1: Lasso for variable selection
    lasso = LassoVariant(method=config["lasso"]["method"], **config["lasso"]["kwargs"])
    lasso.fit(Z_train, d_train)
    selected_features = lasso.selected_features()

    # Use selected features
    Z_selected_train = Z_train[:, selected_features]
    Z_selected_test = Z_test[:, selected_features]

    # In[]: Stage 2: Regression
    reg_model = RegressionModel(method=config["regression"]["method"])
    reg_model.fit(Z_selected_train, y_train)

    # Evaluate the model
    y_pred = reg_model.predict(Z_selected_test)
    mse = mean_squared_error(y_test, y_pred)

    # Results
    print(f"Selected instruments: {selected_features}")
    print(f"Post-Lasso coefficients: {reg_model.coefficients()}")
    print(f"Mean Squared Error: {mse}")
