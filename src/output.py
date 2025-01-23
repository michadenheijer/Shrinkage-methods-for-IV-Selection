import pandas as pd
import numpy as np
import os


def generate_single_output(results, config, save=True):
    """Generates the output for a single estimator specification."""
    # First set output in a dataframe
    output = pd.DataFrame(results)

    # Determine number of instruments equal to 0
    num_instruments_0 = (output["num_selected_instruments"] == 0).sum()

    # Calculate median Bias
    median_bias = output["bias"].median()

    # Calculate median Absolute Deviation
    median_absolute_deviation = output["absolute_deviation"].median()

    # Calculate rejection rate
    rejection_rate = output["reject"].mean()
    
    # Collect some hyperparams
    dataset_design = config["dgp"]["design"]
    n_samples = config["dgp"]["n_samples"]
    n_instruments = config["dgp"]["n_instruments"]
    concentration = config["dgp"]["mu2"]
    
    # Set in dataframe also add seed
    results = {"N(0)": num_instruments_0, "Bias": median_bias, "MAD": median_absolute_deviation, 
               f"rp({config['regression']['alpha']})": rejection_rate, "Seed": config["seed"], "Design": dataset_design,
               "N samples": n_samples, "N instruments": n_instruments, "Concentration": concentration, "Model": config["model_name"],
               "Correlation": config["dgp"]["correlation"]}
    results = pd.DataFrame([results])

    # Save output in a csv file and results folder
    if save:
        # Create results folder if it does not exist
        if not os.path.exists("results"):
            os.makedirs("results")
        
        file_path = f"results/{config['model_name']}.csv"
        if os.path.exists(file_path):
            results.to_csv(file_path, mode='a', header=False, index=False)
        else:
            results.to_csv(file_path, index=False)
            
    return results

    