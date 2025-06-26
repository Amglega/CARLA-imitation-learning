import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_csv", type=str,default="datasets/CARLA_dataset_0/data.csv", help="Data file to plot")

args = parser.parse_args()
# Load the CSV file
try:
    df = pd.read_csv(args.data_csv)
except FileNotFoundError:
    print("Error: file not found")
    exit()

# Check if the columns exist
required_columns = ['throttle', 'steer', 'brake']
if not all(col in df.columns for col in required_columns):
    print(f"Error: Missing one or more columns in the CSV file. Required columns are: {required_columns}")
    exit()

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

# Plot each column in a separate subplot
axes[0].plot(df['throttle'])
axes[0].set_ylabel('Throttle')
axes[0].set_title('Throttle Data')

axes[1].plot(df['steer'])
axes[1].set_ylabel('Steer')
axes[1].set_title('Steer Data')

axes[2].plot(df['brake'])
axes[2].set_ylabel('Brake')
axes[2].set_title('Brake Data')
axes[2].set_xlabel('Data Point Index')

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

# Show the plots
plt.show()
