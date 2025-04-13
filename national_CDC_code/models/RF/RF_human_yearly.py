from sklearn import ensemble, metrics
import pandas as pd
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2] / "data_demo" / "cdc_demo_with_climate.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Clean and convert Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# Select columns after 'Date'
date_index = data.columns.get_loc("Date")
data_pred = data.iloc[:, date_index + 1:].copy()
data_pred["Year"] = data["Year"]
data_pred["Reported human cases"] = data["Reported human cases"]
data_pred["FIPS"] = data["FIPS"]
data_pred = data_pred.reset_index(drop=True)

# Split into training and testing sets
train = data_pred[data_pred["Year"] < 2018].copy()
test = data_pred[data_pred["Year"] >= 2018].copy()

# Impute missing values using FIPS-wise mean from training data
fips_avg = train.groupby("FIPS")["Reported human cases"].mean()
train["Reported human cases"] = train["Reported human cases"].fillna(train["FIPS"].map(fips_avg))
test["Reported human cases"] = test["Reported human cases"].fillna(test["FIPS"].map(fips_avg))

# Drop rows with remaining NaNs
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Separate labels
train_labels = train.pop("Reported human cases").values
test_labels = test.pop("Reported human cases").values

# Remove non-feature columns
train = train.drop(columns=["Year"])
test = test.drop(columns=["Year"])

# Run Random Forest model 100 times
rf_mse_list = []
rf_r2_list = []
for i in range(100):
    rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)
    rf.fit(train, train_labels)
    y_predict = rf.predict(test)

    if i == 0:
        # Save prediction results from the first iteration only
        df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
        df.to_csv(RESULTS_DIR / "human_yearly_cdc_rf.csv", index=False)

    # Compute metrics
    mse = metrics.mean_squared_error(test_labels, y_predict)
    r2 = metrics.r2_score(test_labels, y_predict)

    rf_mse_list.append(mse)
    rf_r2_list.append(r2)

# Print average metrics
print("Average MSE:", sum(rf_mse_list) / len(rf_mse_list))
print("Average R2:", sum(rf_r2_list) / len(rf_r2_list))
