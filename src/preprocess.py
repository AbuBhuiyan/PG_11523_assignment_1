import pandas as pd
import sys

# Input/output paths
input_file = sys.argv[1]
output_file = sys.argv[2]

# Load dataset
df = pd.read_csv(input_file)

# -------------------------
# Handle missing/invalid data
# -------------------------
# Cost
df['cost'].fillna(df['cost'].median(), inplace=True)
df['cost_2'].fillna(df['cost_2'].median(), inplace=True)

# Latitude/Longitude: drop rows <2% missing
df.dropna(subset=['lat', 'lng'], inplace=True)

# Ratings & votes
df['rating_text'].fillna("Unrated", inplace=True)
df['rating_number'].fillna(0, inplace=True)
df['votes'].fillna(0, inplace=True)

# Type
df['type'].fillna("Unknown", inplace=True)

# Save processed data
df.to_csv(output_file, index=False)
print(f"Preprocessing done. Output saved to {output_file}")
