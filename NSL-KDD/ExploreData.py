import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Load the Clean Dataset ---

# Get the directory where this Python script is located
script_dir = Path(__file__).parent

# Define the path to your clean CSV
csv_file = script_dir / 'KDDTrain_with_headers.csv'

print(f"Loading '{csv_file}'...")
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"ERROR: File not found. Make sure 'KDDTrain_with_headers.csv' is in the same folder as this script.")
    exit()

print("Data loaded successfully.")

# --- 2. Analyze the 'label' Column ---

# First, let's see the simple 'normal' vs 'attack' breakdown
# We create a new column 'attack_type'
# It will be 'normal' if the label is 'normal', and 'attack' otherwise
df['attack_type'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

print("\n--- Overall Data Breakdown ---")
print(df['attack_type'].value_counts())
# value_counts() gives you a quick count, e.g.:
# attack    67343
# normal    67342
# (Note: In KDDTrain+, the numbers are almost 50/50, which is unrealistic but good for training)

# --- 3. Analyze the Specific Attack Types ---

# Now, let's look at the *specific* attacks.
# We'll filter out the 'normal' traffic to see the attack labels
attack_df = df[df['attack_type'] == 'attack']

print("\n--- Top 10 Most Common Attacks ---")
attack_counts = attack_df['label'].value_counts()
print(attack_counts.head(10))

# --- 4. Create a Visualization ---
# This is perfect for your presentation slide!

# Get the top 10 attacks for a clean chart
top_10_attacks = attack_counts.head(10)

plt.figure(figsize=(12, 8))  # Make the figure larger
plt.bar(top_10_attacks.index, top_10_attacks.values, color='salmon')
plt.title('Top 10 Attack Types in NSL-KDD Training Set', fontsize=16)
plt.xlabel('Attack Type', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate labels to prevent overlap
plt.tight_layout()  # Adjust layout to fit labels

# Save the chart as an image file
chart_file = script_dir / 'attack_distribution_chart.png'
plt.savefig(chart_file)

print(f"\nSuccessfully saved bar chart to '{chart_file}'")
print("You can now add this PNG image to your presentation!")

# Optional: Show the plot
# plt.show()