# PG-11523 Assignment 1: Restaurant Data Analysis & Predictive Modeling

This repository contains the complete pipeline for analyzing restaurant data, performing exploratory data analysis (EDA), feature engineering, and building regression and classification models.

## Project Structure
PG_11523_assignment_1/
│
├── data/
│ ├── zomato_df_final_data.csv # Raw dataset
│ └── sydney.geojson # Sydney suburbs GeoJSON
│
├── models/
│ └── (saved models via DVC)
│
├── results/
│ ├── plots/ # EDA & model visualization
│ └── evaluation.json # Model evaluation metrics
│
├── src/
│ └── full_pipeline.py # Complete EDA + modeling + plots pipeline
│
├── .gitignore
├── dvc.yaml # DVC pipeline 
├── requirements.txt
└── README.md


### 1. Clone Repository

git clone https://github.com/AbuBhuiyan/PG_11523_assignment_1.git
cd PG_11523_assignment_1

2. Install Dependencies
pip install -r requirements.txt

3. Run Full Pipeline
python src/full_pipeline.py data/zomato_df_final_data.csv results/

This will:

Perform EDA (plots, distribution analysis, correlations, geospatial visualization)

Handle missing values

Conduct feature engineering

Train regression & classification models

Save results & evaluation metrics to the results/ folder

4.DVC
dvc init
dvc add data/zomato_df_final_data.csv
dvc repro

Key Insights

Distribution of restaurants by cuisine, suburb, and type

Cost vs Rating analysis

Skewed distribution of votes

Geospatial density of selected cuisines in Sydney

Model comparison for classification and regression

Author

Abu Bhuiyan
Master of Data Science - Canberra University / Sydney University


