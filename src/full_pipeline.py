# src/full_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import geopandas as gpd
import plotly.express as px
import joblib
import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# -------------------------
# Command-line arguments
# -------------------------
input_file = sys.argv[1]
results_dir = sys.argv[2]
models_dir = sys.argv[3]

# Create folders if they don't exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
if input_file.endswith("restaurant_data.csv"):
    input_file = "data/zomato_df_final_data.csv"

restaurant_data = pd.read_csv(input_file)

# ===============================
# Quick Overview
# ===============================
restaurant_data.describe().to_csv(f"{results_dir}/data_summary.csv")
restaurant_data.isnull().sum().to_csv(f"{results_dir}/missing_values.csv")

# ===============================
# Handle Missing / Invalid Data
# ===============================
median_cost = restaurant_data['cost'].median()
restaurant_data['cost'].fillna(median_cost, inplace=True)
restaurant_data['cost_2'].fillna(median_cost / restaurant_data['cost'].median() * restaurant_data['cost_2'].median(), inplace=True)
restaurant_data.dropna(subset=['lat','lng'], inplace=True)
restaurant_data['rating_text'].fillna("Unrated", inplace=True)
restaurant_data['rating_number'].fillna(0, inplace=True)
restaurant_data['votes'].fillna(0, inplace=True)
restaurant_data['type'].fillna("Unknown", inplace=True)
restaurant_data.to_csv(f"{results_dir}/processed.csv", index=False)

# =============================== 
# EDA Plots
# ===============================
def save_plot(fig, name):
    plt.savefig(f"{results_dir}/{name}.png", bbox_inches="tight")
    plt.close()

# Cost distribution
plt.figure(figsize=(8,5))
sns.histplot(restaurant_data['cost'], bins=20, kde=True)
plt.title("Distribution of Restaurant Cost")
save_plot(plt, "cost_distribution")

# Rating distribution
plt.figure(figsize=(8,5))
sns.histplot(restaurant_data['rating_number'], bins=10, kde=True)
plt.title("Distribution of Ratings")
save_plot(plt, "rating_distribution")

# Type distribution
plt.figure(figsize=(8,5))
sns.countplot(y='type', data=restaurant_data, order=restaurant_data['type'].value_counts().index)
plt.title("Distribution of Restaurant Types")
save_plot(plt, "type_distribution")

# Cost vs Votes
plt.figure(figsize=(8,5))
sns.scatterplot(x='votes', y='cost', data=restaurant_data)
plt.title("Cost vs Votes")
save_plot(plt, "cost_votes_correlation")

# Unique cuisines
plt.figure(figsize=(12,6))
sns.countplot(y='cuisine', data=restaurant_data, order=restaurant_data['cuisine'].value_counts().index)
plt.title("Restaurant Count by Cuisine")
save_plot(plt, "cuisine_distribution")

# Top 3 suburbs
top_suburbs = restaurant_data['subzone'].value_counts().head(3)
plt.figure(figsize=(8,5))
sns.barplot(x=top_suburbs.values, y=top_suburbs.index)
plt.title("Top 3 Suburbs with Most Restaurants")
save_plot(plt, "top_suburbs")

# Cost vs Rating (Poor/Excellent)
g = sns.FacetGrid(restaurant_data, col="rating_text", col_order=['Poor', 'Excellent'],
                  col_wrap=2, height=4, sharex=True, sharey=True)
g.map(sns.histplot, "cost", bins=15, color="skyblue")
g.set_titles("{col_name} Rating")
g.set_axis_labels("Cost (AUD)", "Count")
plt.subplots_adjust(top=0.85)
g.fig.suptitle("Distribution of Restaurant Costs by Rating", fontsize=16)
g.savefig(f"{results_dir}/cost_by_rating.png")
plt.close()

# Static boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='rating_text', y='cost', data=restaurant_data,
            order=['Poor','Average','Good','Very Good','Excellent'], palette="Set2")
plt.title("Restaurant Cost by Rating")
plt.xticks(rotation=30)
save_plot(plt, "cost_boxplot_by_rating")

# Interactive plot (Plotly)
fig = px.box(restaurant_data, x='rating_text', y='cost',
             category_orders={'rating_text': ['Poor','Average','Good','Very Good','Excellent']},
             title="Interactive Restaurant Cost by Rating")
fig.write_html(f"{results_dir}/interactive_cost_boxplot.html")

# ===============================
# Geospatial Analysis (Japanese example)
# ===============================
try:
    gdf = gpd.read_file("sydney.geojson")
    restaurant_data['cuisine'] = restaurant_data['cuisine'].apply(eval)
    df_exploded = restaurant_data.explode('cuisine')
    cuisine_counts = df_exploded.groupby(['subzone','cuisine']).size().reset_index(name="count")
    selected_cuisine = "Japanese"
    cuisine_map = cuisine_counts[cuisine_counts['cuisine']==selected_cuisine]
    merged = gdf.merge(cuisine_map, left_on="SSC_NAME", right_on="subzone", how="left")
    merged['count'] = merged['count'].fillna(0)
    fig, ax = plt.subplots(figsize=(12,10))
    merged.plot(column="count", cmap="OrRd", linewidth=0.8, edgecolor="grey", legend=True,
                legend_kwds={'label': f"{selected_cuisine} restaurants per suburb"}, ax=ax)
    ax.set_title(f"Density of {selected_cuisine} Restaurants in Sydney", fontsize=16)
    ax.axis("off")
    plt.savefig(f"{results_dir}/{selected_cuisine}_choropleth.png")
    plt.close()
except Exception as e:
    print("Geospatial plot skipped:", e)


# ===============================
# Regression Modeling
# ===============================
# ===============================
# Regression Modeling
# ===============================
X = restaurant_data[['votes','cost']]
y_reg = restaurant_data['rating_number']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
pd.DataFrame({"MSE": [mse]}).to_csv(os.path.join(results_dir, "regression_results.csv"), index=False)

# Save regression model + scaler
joblib.dump(reg_model, os.path.join(models_dir, "regression_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "regression_scaler.pkl"))

# ===============================
# Classification Modeling
# ===============================
restaurant_data['high_rated'] = (restaurant_data['rating_number']>=4).astype(int)
y_clf = restaurant_data['high_rated']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(max_iter=500, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save each classification model
    joblib.dump(model, os.path.join(models_dir, f"{name}_model.pkl"))
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    })
pd.DataFrame(results).to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
