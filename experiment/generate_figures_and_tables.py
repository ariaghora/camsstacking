import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from experiment_data_loader import available_datasets

plt.rcParams["text.usetex"] = True

sys.path.append("../")

"""
TABLE: Training results
"""
df_training_result = pd.read_csv("output/training_result.csv", index_col=0)

# Set index name to dataset name
df_training_result.index.name = "Dataset"
df_training_result.index = df_training_result.index.str.replace("_", " ")
df_training_result.index = df_training_result.index.str.title()


# New column names
new_column_names = [
    "CAMSStacker",
    "LinearStacker",
    "LR",
    "DT",
    "NB",
]

# Rename columns
df_training_result.columns = new_column_names

# Convert to percent
df_training_result = df_training_result * 100
df_training_result = df_training_result.round(2)

# Replace max value in each row with \textbf
for row in df_training_result.iterrows():
    max_value = row[1].max()
    idx_max_value = row[1].idxmax()
    df_training_result.loc[row[0], idx_max_value] = (
        r"\textbf{" + f"{max_value:.2f}" + "}"
    )

# save the training result to latex table.
df_training_result.to_latex(
    "report/training_result.tex",
    # float_format="%.2f",
    caption="Training result (in \%)",
    escape=False,
    column_format="l" + "r" * len(new_column_names),
    label="tab:training_result",
)


for dataset in available_datasets:
    clf = joblib.load(f"output/{dataset}_cams_stacker.pkl")
    X_train, X_test, y_train, y_test = joblib.load(f"output/{dataset}_data.pkl")

    weights = clf.weight_estimator.predict(X_test)
    plt.boxplot(weights, labels=new_column_names[2:])
    plt.xticks(ticks=range(1, len(new_column_names) - 1), labels=new_column_names[2:])
    tikzplotlib.save(f"report/{dataset}_weights_distribution.tex")
    plt.close()

# Copy everything inside report/ to ~/research/camstacker/
# This is for manuscript only. You can ignore this.
os.system("cp -r report/* ~/research/camstacker/")
