import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

# -------------------------
# Feature engineering
# -------------------------
# Cuisine diversity
df['cuisine'] = df['cuisine'].apply(eval if isinstance(df['cuisine'].iloc[0], str) else lambda x: x)
df['cuisine_diversity'] = df['cuisine'].apply(len)

# Cost bins
bins = [0, 30, 70, df['cost'].max()]
labels = ['Cheap', 'Mid-range', 'Expensive']
df['cost_bin'] = pd.cut(df['cost'], bins=bins, labels=labels, include_lowest=True)

# Popularity flag
median_votes = df['votes'].median()
df['popular'] = (df['votes'] >= median_votes).astype(int)

# Drop unnecessary columns
drop_cols = ['address', 'phone', 'link', 'title', 'cuisine']
df = df.drop(columns=drop_cols)

# Encode categorical features
df['groupon'] = df['groupon'].astype(int)
df = pd.get_dummies(df, drop_first=True)

# Save features
df.to_csv(output_file, index=False)
print(f"Feature engineering done. Output saved to {output_file}")
