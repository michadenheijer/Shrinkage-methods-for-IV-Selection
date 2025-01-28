import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# First case
# Medium colinearity & 100 instruments
compare_models = ["ElasticNet (BIC)", "ElasticNet (CV)", "Minimax (BIC)", "Minimax (CV)", "Post-Lasso (BIC)", "Post-Lasso (CV)"]

# Choose the metric to visualize: "MAD", "Bias", "rp(0.05)"
metric = "MAD"
n_samples = 100
n_instruments = 250

fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

for i, correlation in enumerate([0.5, 0.9]):
    result_df = pd.DataFrame()
    for concentration in [30, 180]:
        for model in compare_models:
            data = pd.read_csv(f"results/{model}.csv")
            data_filtered = data[(data["Correlation"] == correlation) & (data["N instruments"] == n_instruments) & (data["Concentration"] == concentration) & (data["N samples"] == n_samples)].copy()
            data_filtered.loc[:, "Concentration"] = concentration
            result_df = pd.concat([result_df, data_filtered])
    result_df = result_df.reset_index(drop=True)
    
    # Create bar chart
    bar_width = 0.2
    index = np.arange(4)
    colors = ['b', 'g', 'r']
    model_types = ["ElasticNet", "Minimax", "Post-Lasso"]
    
    for j, model_type in enumerate(model_types):
        means = []
        stds = []
        for concentration in [30, 180]:
            for criterion in ["CV", "BIC"]:
                model = f"{model_type} ({criterion})"
                means.append(result_df[(result_df["Model"] == model) & (result_df["Concentration"] == concentration)][metric].mean())
                stds.append(result_df[(result_df["Model"] == model) & (result_df["Concentration"] == concentration)][metric].std())
        
        axs[i].bar(index + j * bar_width, means, bar_width, yerr=stds, capsize=5, label=model_type, color=colors[j])
    
    #axs[i].set_xlabel('Models')
    axs[i].set_title(f'Correlation = {correlation}')
    axs[i].set_xticks(index + bar_width)
    axs[i].set_xticklabels(['$\mu^2=30$, CV', '$\mu^2=30$, BIC', '$\mu^2=180$, CV', '$\mu^2=180$, BIC'], rotation=45, ha='right')

axs[0].set_ylabel(metric)
axs[0].legend()
fig.suptitle(f'Comparison of Models - N samples = {n_samples}, N instruments = {n_instruments}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"plots/Comparison of Models - N samples = {n_samples}, N instruments = {n_instruments}, Metric = {metric}.png")
plt.show()
