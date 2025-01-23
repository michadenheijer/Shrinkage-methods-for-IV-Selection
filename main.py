# In[]:
import numpy as np
import yaml
from scipy.stats import norm
from joblib import Parallel, delayed
import tqdm
from src.regression import RegressionModel
from src.Lassomethods import LassoVariant
from src.dataset import simulate_dataset
from src.output import generate_single_output

# In[]:
CONFIG_PATH = "configs/postLasso_IC.yaml"


def load_config(config_path):
    """Loads settings from a YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def single_simulation(config, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    data = simulate_dataset(config)

    # In[]: Stage 1: Lasso for variable selection
    lasso = LassoVariant(
        config["dgp"]["n_instruments"],
        config["dgp"]["n_samples"],
        method=config["lasso"]["method"],
        seed=seed,
        **config["lasso"]["kwargs"]
    )
    lasso.fit(data["Z"], data["d"])
    selected_features = lasso.selected_features()

    # Use selected features
    Z_selected = data["Z"][:, selected_features]
    num_selected_instruments = Z_selected.shape[1]

    # If no instruments are selected, return NaNs
    if num_selected_instruments == 0:
        # Use instrument with highest correlation
        correlation = np.abs(np.corrcoef(data["Z"], data["d"], rowvar=False))
        max_corr_idx = np.argmax(correlation[:-1, -1])
        Z_selected = data["Z"][:, max_corr_idx]
        
    # In[]: Stage 2: Regression
    constant = np.ones((len(Z_selected), 1))  # Add constant term (required for 2SLS)
    reg_model = RegressionModel(method=config["regression"]["method"])
    reg_model.fit(
        dependent=data["y"], exog=constant, endog=data["d"], instruments=Z_selected
    )
    reg_coefficients = reg_model.coefficients()

    # Now collect the output
    # For 2SLS
    if config["regression"]["method"] == "2sls":
        bias = reg_coefficients.loc["exog"]
        absolute_deviation = np.abs(
            reg_coefficients.loc["endog"] - config["dgp"]["beta_true"]
        )
        z_value = (
            reg_coefficients.loc["endog"] - config["dgp"]["beta_true"]
        ) / reg_model.model.std_errors.loc["endog"]
        p_value = 2 * (1 - norm.cdf(np.abs(z_value)))
        reject = p_value < config["regression"]["alpha"]

    elif config["regression"]["method"] == "fuller":
        raise NotImplementedError("Fuller method not implemented")
        # bias = reg_coefficients.loc["exog"]
        # absolute_deviation = np.abs(reg_coefficients.loc["endog"] - config["dgp"]["beta_true"])
        # p_values = reg_model.p_values()
        # reject = p_values.loc["endog"] < config["regression"]["alpha"]

    output = {
        "num_selected_instruments": num_selected_instruments,
        "bias": bias,
        "absolute_deviation": absolute_deviation,
        "reject": reject,
    }

    return output


# In[]:

# Example usage
if __name__ == "__main__":
    # In[]: # Load configuration
    config = load_config(CONFIG_PATH)
    seed = None if config["seed"] == "None" else config["seed"]
    num_simulations = config["num_simulations"]
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Lets loop over all different settings (not very clean)
    for n_samples in [250, 100]:
        config["dgp"]["n_samples"] = n_samples
        
        for n_instruments in [100, 250]:
            if n_samples == 250 and n_instruments == 250:
                continue
            config["dgp"]["n_instruments"] = n_instruments
        
            for design in ["Exponential", 5, 50]:
                config["dgp"]["design"] = design
                
                for mu2 in [30, 180]:
                    config["dgp"]["mu2"] = mu2
                    
                    #for corr in [0.5, 0.9]:
                    for corr in [0.9]:
                        config["dgp"]["correlation"] = corr
    
                        print(f"Running simulations for: {n_samples} samples, {n_instruments} instruments, {design} design, mu2={mu2}")
                        # Run simulations using joblib for parallel processing
                        results = Parallel(n_jobs=-config["n_cores"])(delayed(single_simulation)(config, seed) for _ in tqdm.tqdm(range(num_simulations)))
                        
                        # Generate output
                        output = generate_single_output(results, config)
                        print(output)
    
    
    
    

