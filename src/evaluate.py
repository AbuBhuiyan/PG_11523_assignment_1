import pickle
import sys
import json
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

reg_model_file = sys.argv[1]
clf_model_file = sys.argv[2]
output_file = sys.argv[3]

# Load models
with open(reg_model_file, 'rb') as f:
    reg_model = pickle.load(f)
with open(clf_model_file, 'rb') as f:
    clf_model = pickle.load(f)

# Dummy evaluation using same feature CSV (for simplicity)
df_features = 'data/features.csv'
df = pd.read_csv(df_features)

# Regression
X_reg = df.drop(columns=['rating_number'])
y_reg = df['rating_number']

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_test = imputer.fit_transform(X_test)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

y_pred_reg = reg_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# Classification
y_clf = (df['rating_number'] >= 3).astype(int)
X_clf = df.drop(columns=['rating_number'])

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

X_test = imputer.fit_transform(X_test)
X_test_scaled = scaler.fit_transform(X_test)

y_pred_clf = clf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_clf)

# Save evaluation results
results = {
    "regression_rmse": rmse,
    "classification_accuracy": accuracy
}

with open(output_file, 'w') as f:
    json.dump(results, f)

print(f"Evaluation completed. Results saved to {output_file}")
