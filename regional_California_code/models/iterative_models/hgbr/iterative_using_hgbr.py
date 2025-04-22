from pathlib import Path
from sklearn import ensemble, metrics
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp
import seaborn as sns

# ====== Set correct paths ======
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[3] / "data_demo" / "CA_human_data_2004_to_2023_final_all_counties_CDPH_scraped.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ====== Load dataset ======
data = pd.read_csv(DATA_PATH, index_col=False, header=0)

# Drop columns that are not features and target
data = data.drop([
    "Date", "County", "Latitude", "Longitude",
    "Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count",
    "Cx tarsalis", "Cx pipiens", "ONI"
], axis=1, errors="ignore")

# Drop empty or constant columns
data = data.dropna(axis=1, how='all')
data = data.loc[:, data.var() != 0]
data = data.reset_index(drop=True)

# Print zero-variance columns
print(data.columns[data.var() == 0])

# Fill missing Human_Disease_Count with 0
data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)

# ====== Prepare main variables ======
years = data["Year"].unique()
years.sort()
tuning_years_list = list(range(2005, 2023))

tuning_year_dict = {}
best_hyperparameters_df = pd.DataFrame(columns=[
    "tuning_year", "max_depth", "max_iter", "learning_rate", "l2_regularization",
    "max_leaf_nodes", "min_samples_leaf", "max_bins", "scoring"
])

# ====== Loop over tuning years ======
for tuning_year in tuning_years_list:
    print(f"Tuning year: {tuning_year}")
    best_hyperparameters = []

    for year in years:
        if year != tuning_year:
            continue

        train = data[data['Year'] < year].copy()
        test = data[data['Year'] == year].copy()

        train = train.dropna(subset=["Human_Disease_Count"]).reset_index(drop=True)
        test = test.dropna(subset=["Human_Disease_Count"]).reset_index(drop=True)

        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train = train.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")
        test = test.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")

        scoring_map = ['loss', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        def objective(params):
            params['max_depth'] = int(params['max_depth'])
            params['max_iter'] = int(params['max_iter'])
            params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
            params['max_bins'] = int(params['max_bins'])

            hgbr = ensemble.HistGradientBoostingRegressor(**params)
            hgbr.fit(train, train_labels)
            y_predict = hgbr.predict(test)
            return -metrics.r2_score(test_labels, y_predict)

        space = {
            'max_depth': hp.quniform('max_depth', 1, 30, 1),
            'max_iter': hp.quniform('max_iter', 100, 1000, 100),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
            'l2_regularization': hp.uniform('l2_regularization', 0.0, 1.0),
            'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 10),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
            'max_bins': hp.quniform('max_bins', 10, 255, 5),
            'scoring': hp.choice('scoring', ['loss', 'neg_mean_squared_error', 'neg_mean_absolute_error'])
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

        best['scoring'] = scoring_map[int(best['scoring'])]
        best['max_depth'] = int(best['max_depth'])
        best['max_iter'] = int(best['max_iter'])
        best['max_leaf_nodes'] = int(best['max_leaf_nodes'])
        best['min_samples_leaf'] = int(best['min_samples_leaf'])
        best['max_bins'] = int(best['max_bins'])

        best_hyperparameters.append(best)

        best_hyperparameters_df = best_hyperparameters_df.append({
            "tuning_year": year,
            "max_depth": best["max_depth"],
            "max_iter": best["max_iter"],
            "learning_rate": best["learning_rate"],
            "l2_regularization": best["l2_regularization"],
            "max_leaf_nodes": best["max_leaf_nodes"],
            "min_samples_leaf": best["min_samples_leaf"],
            "max_bins": best["max_bins"],
            "scoring": best["scoring"]
        }, ignore_index=True)

    print("Best hyperparameters:", best_hyperparameters)

    rmse_list = []
    r2_list = []

    predict_year = 2005
    for year in years:
        if year < predict_year:
            continue

        train = data[data['Year'] < year].copy()
        test = data[data['Year'] == year].copy()

        train = train.dropna(subset=["Human_Disease_Count"]).reset_index(drop=True)
        test = test.dropna(subset=["Human_Disease_Count"]).reset_index(drop=True)

        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train = train.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")
        test = test.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")

        hgbr = ensemble.HistGradientBoostingRegressor(**best_hyperparameters[0])
        hgbr.fit(train, train_labels)

        y_predict = hgbr.predict(test)

        mse = metrics.mean_squared_error(test_labels, y_predict)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(test_labels, y_predict)

        rmse_list.append(rmse)
        r2_list.append(r2)

        predict_year += 1
        print(f"predict year: {year}, Q2: {r2}")

    tuning_year_dict[tuning_year] = {"q2": r2_list, "rmse": rmse_list}

# Save best hyperparameters
best_hyperparameters_df.to_csv(RESULTS_DIR / "hyperparameter_tuning_hgbr_impute_0_mos_abundance.csv", index=False)

# ====== Plot heatmaps ======
x_axis_years = list(range(2005, 2024))

# Plot Q2 heatmap
fig, ax = plt.subplots(figsize=(30, 25))
q2_array = [tuning_year_dict[t]["q2"] for t in tuning_years_list]
q2_array = np.clip(q2_array, -1, 1)
mask = np.tril(np.ones_like(q2_array), k=0).astype(bool)
q2_array = np.where(mask, np.nan, q2_array)

sns.heatmap(q2_array, cmap='RdBu', annot=False, linewidths=0.5, ax=ax, center=0, vmax=1, vmin=-1)

cbar = ax.collections[0].colorbar
cbar.set_label("Q2", fontsize=30)
cbar.ax.tick_params(labelsize=25)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

ax.set_xlabel("Predicting Year", fontsize=30)
ax.set_ylabel("HGBR Model Using Best Hyperparameter for Predicting Year ____", fontsize=30)

ax.set_xticks(np.arange(len(x_axis_years)) + 0.5)
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5)
ax.set_xticklabels(x_axis_years, fontsize=25, rotation=45)
ax.set_yticklabels(tuning_years_list, fontsize=25)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "q2_tuning_year_heatmap_hgbr_mos_abundance.png", dpi=300)
plt.close()

# Plot RMSE heatmap
fig, ax = plt.subplots(figsize=(30, 25))
rmse_array = [tuning_year_dict[t]["rmse"] for t in tuning_years_list]

sns.heatmap(rmse_array, cmap='Reds', annot=False, linewidths=0.5, ax=ax)

ax.set_xlabel("Predicting Year", fontsize=22)
ax.set_ylabel("HGBR Model Using Best Hyperparameter for Predicting Year ____", fontsize=22)
ax.set_title("RMSE Heatmap for HGBR Hyperparameter Tuning", fontsize=25)

ax.set_xticks(np.arange(len(x_axis_years)) + 0.5)
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5)
ax.set_xticklabels(x_axis_years, fontsize=18)
ax.set_yticklabels(tuning_years_list, fontsize=18)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "rmse_tuning_year_heatmap_hgbr_mos_abundance.png", dpi=300)
plt.close()