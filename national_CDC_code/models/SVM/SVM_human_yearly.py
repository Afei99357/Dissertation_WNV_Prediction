from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
from pathlib import Path

# Define base and data paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2] / "data_demo" / "cdc_demo_with_climate.csv"
RESULTS_PATH = BASE_DIR / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Clean and convert Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# Select columns after 'Date' as predictors
date_index = data.columns.get_loc("Date")
data_pred = data.iloc[:, date_index + 1:].copy()
data_pred["Year"] = data["Year"]
data_pred["Reported human cases"] = data["Reported human cases"]
data_pred["FIPS"] = data["FIPS"]
data_pred = data_pred.reset_index(drop=True)

# Split into training and testing sets
train = data_pred[data_pred["Year"] < 2018].copy()
test = data_pred[data_pred["Year"] >= 2018].copy()

# Impute missing values using FIPS-wise mean from training set
fips_avg = train.groupby("FIPS")["Reported human cases"].mean()
train["Reported human cases"] = train["Reported human cases"].fillna(train["FIPS"].map(fips_avg))
test["Reported human cases"] = test["Reported human cases"].fillna(test["FIPS"].map(fips_avg))

# Drop rows with any remaining NaN values
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Separate labels from features
train_labels = train.pop("Reported human cases").values
test_labels = test.pop("Reported human cases").values

# Remove Year column
train = train.drop(columns=["Year"])
test = test.drop(columns=["Year"])

# Normalize features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Train SVM model
clf = SVR(epsilon=0.3, gamma=0.002, kernel="rbf", C=100)
clf.fit(train_scaled, train_labels)
y_predict = clf.predict(test_scaled)

# Save predictions to CSV
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv(RESULTS_PATH / "human_yearly_cdc_svm.csv", index=False)

# Calculate metrics
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)

# Baseline model using mean of training labels
baseline_preds = [train_labels.mean()] * len(test_labels)
fake_model_mse = metrics.mean_squared_error(test_labels, baseline_preds)
fake_model_r2 = metrics.r2_score(test_labels, baseline_preds)

# Print evaluation results
print("The mean squared error of SVM Model vs. Baseline Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The R2 score of SVM Model vs. Baseline Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
