import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "iteration_count_results.csv" # Change this to your file path
df = pd.read_csv(csv_file)

# Extract the experiment types and run data
experiment_types = df["experiment_type"]
run_data = df.iloc[:, 1:]  # Exclude the first column ("experiment_type")

# Calculate the mean and standard deviation for each experiment
means = run_data.mean(axis=1)
std_devs = run_data.std(axis=1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(experiment_types, means, yerr=std_devs, capsize=5, alpha=0.75, color="skyblue", edgecolor="black")

# Formatting the plot
plt.xlabel("Experiment Type")
plt.ylabel("Average Run Length")
plt.title("Average Run Length with Standard Deviation")
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
