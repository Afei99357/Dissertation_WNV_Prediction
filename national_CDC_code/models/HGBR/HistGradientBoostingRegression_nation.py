from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import metrics
import pandas as pd
import numpy as np
from pathlib import Path

# Set base and data paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2] / "data_demo" / "cdc_demo_with_climate.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Clean and convert Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# Select columns after 'Date'
date_index = data.columns.get_loc("Date")
data_pred = data.iloc[:, date_index + 1:].copy()
data_pred['Year'] = data['Year']
data_pred["Reported human cases"] = data["Reported human cases"]
data_pred["FIPS"] = data["FIPS"]
data_pred = data_pred.reset_index(drop=True)

# Split into training and testing sets
train = data_pred[data_pred["Year"] < 2018].copy()
test = data_pred[data_pred["Year"] >= 2018].copy()

# Fill missing values using FIPS average from training data
fips_avg = train.groupby("FIPS")["Reported human cases"].mean()
train["Reported human cases"] = train["Reported human cases"].fillna(train["FIPS"].map(fips_avg))
test["Reported human cases"] = test["Reported human cases"].fillna(test["FIPS"].map(fips_avg))

# Drop rows with NaNs
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Separate features and labels
train_labels = train.pop("Reported human cases").values
test_labels = test.pop("Reported human cases").values
train = train.drop(columns=["Year"])
test = test.drop(columns=["Year"])

# Train Histogram-based Gradient Boosting Regressor
est = HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)
est.fit(train, train_labels)
y_predict = est.predict(test)

# Save predictions to CSV
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv(RESULTS_DIR / "hgbr_cali_multi_Years_result.csv", index=False)

# Compute evaluation metrics
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)

# Baseline metrics using mean predictor
baseline_preds = [train_labels.mean()] * len(test_labels)
fake_model_mse = metrics.mean_squared_error(test_labels, baseline_preds)
fake_model_r2 = metrics.r2_score(test_labels, baseline_preds)

# Print results
print("The mean squared error of HGBR vs. baseline: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The R2 score of HGBR vs. baseline: {:.03}, vs {:.03}".format(r2, fake_model_r2))
