import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

# Binary classification: rating >= 3
y = (df['rating_number'] >= 3).astype(int)
X = df.drop(columns=['rating_number'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Impute & scale
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model
with open(output_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Classification model trained and saved to {output_file}")
