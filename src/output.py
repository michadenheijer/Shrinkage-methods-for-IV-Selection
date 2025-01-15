import pandas as pd
import numpy as np


def generate_single_output(results, config):
    """Generates the output for a single estimator specification."""
    # First set output in a dataframe
    output = pd.DataFrame(results)
    
    # Determine number of instruments equal to 0
    num_instruments = np.array([result["num_instruments"] for result in results])
    num_instruments_0 = np.sum(num_instruments == 0)
    
    # Calculate median Bias
    median_bias = output["bias"].median()
    
    # Calculate median Absolute Deviation
    median_absolute_deviation = output["absolute_deviation"].median()
    
    # Calculate rejection rate
    rejection_rate = output["reject"].mean()
    
    # Set in dataframe and return
    return pd.DataFrame({"N(0)": num_instruments_0, "Bias": median_bias, "MAD": median_absolute_deviation, f"rp({config["regression"]["alpha"]})": rejection_rate}, index=[0])
    