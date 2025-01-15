# In[]:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.regression import RegressionModel
from src.Lassomethods import LassoVariant
from src.dataset import simulate_dataset
from src.output import generate_single_output
import tqdm
from joblib import Parallel, delayed
import yaml

# In[]:
CONFIG_PATH = "configs/configJasper.yaml"


def load_config(config_path):
    """Loads settings from a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def single_simulation(config, seed=None):
    # Generate data
    data = simulate_dataset(config, seed)

    # In[]: Stage 1: Lasso for variable selection
    lasso = LassoVariant(method=config["lasso"]["method"], seed=seed, **config["lasso"]["kwargs"])
    lasso.fit(data["Z"], data["d"])
    selected_features = lasso.selected_features()

    # Use selected features
    Z_selected = data["Z"][:, selected_features]
    num_selected_instruments = Z_selected.shape[1]
    

    # In[]: Stage 2: Regression
    constant = np.ones((len(Z_selected), 1))  # Add constant term (required for 2SLS)
    reg_model = RegressionModel(method=config["regression"]["method"])
    reg_model.fit(dependent=data["y"], exog=constant, endog=data["d"], instruments=Z_selected)
    reg_coefficients = reg_model.coefficients()
    
    # Now collect the output
    # For 2SLS
    if config["regression"]["method"] == "2sls":
        bias = reg_coefficients.loc["exog"]
        absolute_deviation = np.abs(reg_coefficients.loc["endog"] - config["dgp"]["beta_true"])
        p_values = reg_model.p_values()
        reject = p_values.loc["endog"] < config["regression"]["alpha"]
        
    elif config["regression"]["method"] == "fuller":
        raise NotImplementedError("Fuller method not implemented")
        # bias = reg_coefficients.loc["exog"]
        # absolute_deviation = np.abs(reg_coefficients.loc["endog"] - config["dgp"]["beta_true"])
        # p_values = reg_model.p_values()
        # reject = p_values.loc["endog"] < config["regression"]["alpha"]
        
    output = {"num_selected_instruments": num_selected_instruments, "bias": bias, "absolute_deviation": absolute_deviation, "reject": reject}
        
    return output
    
    
# In[]:
def generate_single_output(results, config):
    """Generates the output for a single estimator specification."""
    # First set output in a dataframe
    output = pd.DataFrame(results)
    
    
    
    # Determine number of instruments equal to 0
    num_instruments = np.array([result["num_instruments"] for result in results])
    num_instruments_0 = np.sum(num_instruments == 0)
    
    # Calculate medi



# Example usage
if __name__ == "__main__":
    # In[]: # Load configuration
    config = load_config(CONFIG_PATH)
    seed = None if config["seed"] == "None" else config["seed"]
    num_simulations = config["num_simulations"]
    if num_simulations != 1 and seed is not None:
        raise ValueError("Multiple simulations require seed to be None.")
    
    # Run simulations using joblib for parallel processing
    results = Parallel(n_jobs=-config["n_cores"])(delayed(single_simulation)(config, seed) for _ in tqdm.tqdm(range(num_simulations)))
    
    # Generate output
    output = generate_single_output(results, config)
    
    
    
    

