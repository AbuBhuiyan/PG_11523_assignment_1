import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

# Features & target
target_col = 'rating_number'
X = df.drop(columns=[target_col])
y = df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute & scale
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model
with open(output_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Regression model trained and saved to {output_file}")
