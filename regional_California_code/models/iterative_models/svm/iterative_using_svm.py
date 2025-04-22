from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# Set paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2]  / "data_demo" / "CA_human_data_2004_to_2023_final_all_counties_CDPH_scraped.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
data = pd.read_csv(DATA_PATH, index_col=False)

# Drop unnecessary columns
DROP_COLUMNS = ["Date", "County", "Latitude", "Longitude", "Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count"]
data = data.drop(columns=DROP_COLUMNS, errors='ignore')

# Drop constant or empty columns
data = data.dropna(axis=1, how='all')
data = data.loc[:, data.var() != 0]
data = data.reset_index(drop=True)

# Fill missing target values
data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)

# Setup tuning years and results holders
tuning_years_list = list(range(2005, 2023))
tuning_year_dict = {}
best_hyperparameters_df = pd.DataFrame(columns=["tuning_year", "C", "epsilon", "kernel", "gamma"])

# Define search space
kernel_map = ["rbf", "linear", "poly", "sigmoid"]
gamma_map = ["scale", "auto"]

for tuning_year in tuning_years_list:
    print(f"Tuning year: {tuning_year}")
    best_hyperparameters = []

    # Train/validation split for hyperparameter tuning
    for year in data['Year'].unique():
        if year != tuning_year:
            continue

        train = data[data['Year'] < year].dropna().reset_index(drop=True)
        test = data[data['Year'] == year].dropna().reset_index(drop=True)

        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train = train.drop(columns=["Month", "FIPS", "Year"], errors='ignore')
        test = test.drop(columns=["Month", "FIPS", "Year"], errors='ignore')

        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)

        def objective(params):
            model = SVR(**params)
            model.fit(train, train_labels)
            y_pred = model.predict(test)
            y_pred[y_pred < 0] = 0
            return -metrics.r2_score(test_labels, y_pred)

        space = {
            "C": hp.uniform("C", 0.1, 10),
            "epsilon": hp.uniform("epsilon", 0.1, 5),
            "kernel": hp.choice("kernel", kernel_map),
            "gamma": hp.choice("gamma", gamma_map)
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200)

        # Define the mappings for kernel and gamma
        kernel_map = ["rbf", "linear", "poly", "sigmoid"]
        gamma_map = ["scale", "auto"]

        # Map the best indices back to the corresponding string values
        best['kernel'] = kernel_map[int(best['kernel'])]
        best['gamma'] = gamma_map[int(best['gamma'])]

        best_hyperparameters.append(best)
        best_hyperparameters_df = pd.concat([
            best_hyperparameters_df,
            pd.DataFrame({"tuning_year": [year], "C": [best["C"]], "epsilon": [best["epsilon"]], "kernel": [best["kernel"]], "gamma": [best["gamma"]]})
        ], ignore_index=True)

    # Iterative prediction using tuned hyperparameters
    rmse_list, r2_list = [], []
    predict_year = 2005

    for year in data['Year'].unique():
        if year < predict_year:
            continue

        train = data[data['Year'] < year].dropna().reset_index(drop=True)
        test = data[data['Year'] == year].dropna().reset_index(drop=True)

        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train = train.drop(columns=["Month", "FIPS", "Year"], errors='ignore')
        test = test.drop(columns=["Month", "FIPS", "Year"], errors='ignore')

        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)

        model = SVR(**best_hyperparameters[0])
        model.fit(train, train_labels)

        y_pred = model.predict(test)
        y_pred[y_pred < 0] = 0

        mse = metrics.mean_squared_error(test_labels, y_pred)
        rmse_list.append(np.sqrt(mse))
        r2_list.append(metrics.r2_score(test_labels, y_pred))

        predict_year += 1

    tuning_year_dict[tuning_year] = {"q2": r2_list, "rmse": rmse_list}

# Save best hyperparameters
best_hyperparameters_df.to_csv(RESULTS_DIR / "best_svm_hyperparameters.csv", index=False)

# Setup heatmaps
x_axis_years = list(range(2005, 2024))

fig, ax = plt.subplots(figsize=(30, 25))
q2_array = np.clip([tuning_year_dict[y]['q2'] for y in tuning_years_list], -1, 1)
mask = np.tril(np.ones_like(q2_array, dtype=bool), k=0)
sns.heatmap(q2_array, cmap='RdBu', mask=mask, center=0, linewidths=0.5, ax=ax)
cbar = ax.collections[0].colorbar
cbar.set_label("Q2", fontsize=30)
cbar.ax.tick_params(labelsize=25)

ax.set_xlabel("Predicting Year", fontsize=30)
ax.set_ylabel("Tuning Year", fontsize=30)
ax.set_xticks(np.arange(len(x_axis_years)) + 0.5)
ax.set_xticklabels(x_axis_years, fontsize=25)
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5)
ax.set_yticklabels(tuning_years_list, fontsize=25)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "q2_tuning_year_heatmap.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(30, 25))
rmse_array = [tuning_year_dict[y]['rmse'] for y in tuning_years_list]
sns.heatmap(rmse_array, cmap='Reds', linewidths=0.5, ax=ax)

ax.set_xlabel("Predicting Year", fontsize=30)
ax.set_ylabel("Tuning Year", fontsize=30)
ax.set_xticks(np.arange(len(x_axis_years)) + 0.5)
ax.set_xticklabels(x_axis_years, fontsize=25)
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5)
ax.set_yticklabels(tuning_years_list, fontsize=25)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "rmse_tuning_year_heatmap.png", dpi=300)
plt.close()
