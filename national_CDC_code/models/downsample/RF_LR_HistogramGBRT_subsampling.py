from sklearn import ensemble, metrics
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2] / "data_demo" / "cdc_demo_with_climate.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Clean Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# Prepare predictors
date_index = data.columns.get_loc("Date")
data_pred = data.iloc[:, date_index + 1:].copy()
data_pred['Year'] = data['Year']
data_pred["FIPS"] = data["FIPS"]
data_pred["Reported human cases"] = data["Reported human cases"]

# Define subsampling function
def balance_classes(data, target_column, seed=123):
    data_majority = data[data[target_column] == 0]
    data_minority = data[data[target_column] > 0]
    if len(data_majority) > len(data_minority):
        data_majority_downsampled = resample(data_majority, replace=False, n_samples=len(data_minority), random_state=seed)
        data_balanced = pd.concat([data_majority_downsampled, data_minority])
    else:
        data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=seed)
        data_balanced = pd.concat([data_minority_upsampled, data_majority])
    return data_balanced

# Train/test split
train = data_pred[data_pred["Year"] < 2018].copy()
test = data_pred[data_pred["Year"] >= 2018].copy()

# Balance training data
train = balance_classes(train, "Reported human cases")

# Fill missing values using FIPS averages
fips_avg = train.groupby("FIPS")["Reported human cases"].mean()
train["Reported human cases"] = train["Reported human cases"].fillna(train["FIPS"].map(fips_avg))
test["Reported human cases"] = test["Reported human cases"].fillna(test["FIPS"].map(fips_avg))

# Drop NaNs
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Extract labels and drop label/time columns
train_labels = train.pop("Reported human cases").values
test_labels = test.pop("Reported human cases").values
train = train.drop(columns=["Year"])
test = test.drop(columns=["Year"])

# Track performance metrics
rf_mse_list = []
rf_r2_list = []
est_mse_list = []
est_r2_list = []

# Run ensemble models
for i in range(100):
    rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)
    rf.fit(train, train_labels)
    y_predict = rf.predict(test)

    est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)
    est.fit(train, train_labels)
    y_predict_est = est.predict(test)

    if i == 0:
        pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict}).to_csv(RESULTS_DIR / "human_yearly_cdc_rf_downsampling.csv", index=False)
        pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict_est}).to_csv(RESULTS_DIR / "human_yearly_cdc_hgbr_downsampling.csv", index=False)

    rf_mse_list.append(metrics.mean_squared_error(test_labels, y_predict))
    rf_r2_list.append(metrics.r2_score(test_labels, y_predict))
    est_mse_list.append(metrics.mean_squared_error(test_labels, y_predict_est))
    est_r2_list.append(metrics.r2_score(test_labels, y_predict_est))

# Print average metrics
print("Average RF MSE:", sum(rf_mse_list)/len(rf_mse_list))
print("Average RF R2:", sum(rf_r2_list)/len(rf_r2_list))
print("HistogramGBRT Average MSE:", sum(est_mse_list)/len(est_mse_list))
print("HistogramGBRT Average R2:", sum(est_r2_list)/len(est_r2_list))

# Linear regression
scaler = StandardScaler()
lr_train = scaler.fit_transform(train)
lr_test = scaler.transform(test)

linear_regression = LinearRegression()
linear_regression.fit(lr_train, train_labels)
y_predict_lr = linear_regression.predict(lr_test)

lr_mse = metrics.mean_squared_error(test_labels, y_predict_lr)
lr_r2 = metrics.r2_score(test_labels, y_predict_lr)

print("Linear Regression MSE:", lr_mse)
print("Linear Regression R2:", lr_r2)

# Plot and save boxplot of all metrics
plt.boxplot([
    rf_mse_list, rf_r2_list,
    [lr_mse], [lr_r2],
    est_mse_list, est_r2_list
], labels=["rf_mse", "rf_r2", "lr_mse", "lr_r2", "est_mse", "est_r2"])

plt.savefig(RESULTS_DIR / "human_yearly_cdc_rf_downsampling_boxplot.png")
