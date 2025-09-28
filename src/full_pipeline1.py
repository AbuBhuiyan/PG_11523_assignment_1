# src/full_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import geopandas as gpd
import plotly.express as px
import os

# -------------------------
# Command-line arguments
# -------------------------
input_file = sys.argv[1]    # raw dataset CSV
output_dir = sys.argv[2]    # directory to save results, plots, models
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
restaurant_data = pd.read_csv(input_file)

# ===============================
# Quick Overview
# ===============================
restaurant_data.info()
restaurant_data.describe()
restaurant_data.isnull().sum()

# Save quick overview to CSV
restaurant_data.describe().to_csv(f"{output_dir}/data_summary.csv")

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

# Save processed CSV
restaurant_data.to_csv(f"{output_dir}/processed.csv", index=False)

# ===============================
# EDA Plots
# ===============================

# ===============================
# Analysis 1: Unique Cuisines
# ===============================
unique_cuisines = restaurant_data['cuisine'].nunique()
print(f"Number of unique cuisines: {unique_cuisines}")
plt.figure(figsize=(12,6))
sns.countplot(y='cuisine', data=restaurant_data, order=restaurant_data['cuisine'].value_counts().index)
plt.title("Restaurant Count by Cuisine")
plt.savefig(f"{output_dir}/cuisine_distribution.png")
plt.close()

# ===============================
# Analysis 2: Top 3 Suburbs
# ===============================
top_suburbs = restaurant_data['subzone'].value_counts().head(3)
print("Top 3 suburbs with most restaurants:\n", top_suburbs)
plt.figure(figsize=(8,5))
sns.barplot(x=top_suburbs.values, y=top_suburbs.index)
plt.title("Top 3 Suburbs with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("Suburb")
plt.savefig(f"{output_dir}/top_suburbs.png")
plt.close()

# ===============================
# Analysis 3: Cost vs Rating
# ===============================
g = sns.FacetGrid(
    restaurant_data, 
    col="rating_text", 
    col_order=['Poor', 'Excellent'],
    col_wrap=2,
    height=4,
    sharex=True,
    sharey=True
)
g.map(sns.histplot, "cost", bins=15, color="skyblue")
g.set_titles("{col_name} Rating")
g.set_axis_labels("Cost (AUD)", "Count")
plt.subplots_adjust(top=0.85)
g.fig.suptitle("Distribution of Restaurant Costs by Rating", fontsize=16)
plt.savefig(f"{output_dir}/cost_by_rating.png")
plt.close()
plt.figure(figsize=(8,5))
sns.histplot(restaurant_data['cost'], bins=20, kde=True)
plt.title("Distribution of Restaurant Cost")
plt.savefig(f"{output_dir}/cost_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(restaurant_data['rating_number'], bins=10, kde=True)
plt.title("Distribution of Ratings")
plt.savefig(f"{output_dir}/rating_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(y='type', data=restaurant_data, order=restaurant_data['type'].value_counts().index)
plt.title("Distribution of Restaurant Types")
plt.savefig(f"{output_dir}/type_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(x='votes', y='cost', data=restaurant_data)
plt.title("Cost vs Votes")
plt.savefig(f"{output_dir}/cost_votes_correlation.png")
plt.close()

# ===============================
# Geospatial Analysis (Japanese cuisine example)
# ===============================
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
plt.savefig(f"{output_dir}/{selected_cuisine}_choropleth.png")
plt.close()

# ===============================
# Static boxplot of cost by rating
# ===============================
plt.figure(figsize=(10,6))
sns.boxplot(x='rating_text', y='cost', data=restaurant_data,
            order=['Poor','Average','Good','Very Good','Excellent'], palette="Set2")
plt.title("Restaurant Cost by Rating", fontsize=16)
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Cost (AUD)", fontsize=12)
plt.xticks(rotation=30)
plt.savefig(f"{output_dir}/cost_boxplot_by_rating.png")
plt.close()

# ===============================
# Interactive visualization (Plotly)
# ===============================
fig = px.box(restaurant_data, x='rating_text', y='cost',
             category_orders={'rating_text': ['Poor','Average','Good','Very Good','Excellent']},
             title="Interactive Restaurant Cost by Rating")
fig.write_html(f"{output_dir}/interactive_cost_boxplot.html")


# ===============================
# Feature Engineering
# ===============================
# Convert cuisine string lists to actual lists
restaurant_data['cuisine'] = restaurant_data['cuisine'].apply(lambda x: eval(x) if isinstance(x,str) else x)
restaurant_data['cuisine_diversity'] = restaurant_data['cuisine'].apply(len)

bins = [0,30,70,restaurant_data['cost'].max()]
labels = ['Cheap','Mid-range','Expensive']
restaurant_data['cost_bin'] = pd.cut(restaurant_data['cost'], bins=bins, labels=labels, include_lowest=True)

median_votes = restaurant_data['votes'].median()
restaurant_data['popular'] = (restaurant_data['votes'] >= median_votes).astype(int)

drop_cols = ['address','phone','link','title','cuisine']
restaurant_data_model = restaurant_data.drop(columns=drop_cols)

restaurant_data_model['groupon'] = restaurant_data_model['groupon'].astype(int)
cat_cols = restaurant_data_model.select_dtypes(include=['object','category']).columns.tolist()
restaurant_data_model_final = pd.get_dummies(restaurant_data_model, columns=cat_cols, drop_first=True)

numeric_cols = ['cost','cost_2','lat','lng','votes','cuisine_diversity']
scaler = StandardScaler()
restaurant_data_model_final[numeric_cols] = scaler.fit_transform(restaurant_data_model_final[numeric_cols])

target_col = 'rating_number'
feature_cols = [col for col in restaurant_data_model_final.columns if col != target_col]
X = restaurant_data_model_final[feature_cols]
y = restaurant_data_model_final[target_col]

# Remove duplicates
df_model = X.copy()
df_model[target_col] = y
df_model = df_model.drop_duplicates()
X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# ===============================
# Regression Models
# ===============================
mask = ~y.isna()
X_reg = X[mask]
y_reg = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

lin_reg = LinearRegression()
lin_reg.fit(X_train_pca, y_train)
y_pred_lr = lin_reg.predict(X_test_pca)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Save regression predictions
pd.DataFrame({"y_true": y_test, "y_pred_lr": y_pred_lr}).to_csv(f"{output_dir}/regression_predictions.csv", index=False)

print(f"Linear Regression MSE: {mse_lr:.4f}")

# ===============================
# Classification Models
# ===============================
y_binary = (y >= 3).astype(int)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

imputer_clf = SimpleImputer(strategy="mean")
X_train_clf = imputer_clf.fit_transform(X_train_clf)
X_test_clf = imputer_clf.transform(X_test_clf)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosted Trees": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_clf_scaled, y_train_clf)
    y_pred = model.predict(X_test_clf_scaled)

    precision = precision_score(y_test_clf, y_pred)
    recall = recall_score(y_test_clf, y_pred)
    f1 = f1_score(y_test_clf, y_pred)
    acc = accuracy_score(y_test_clf, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/classification_results.csv", index=False)
print(results_df)
