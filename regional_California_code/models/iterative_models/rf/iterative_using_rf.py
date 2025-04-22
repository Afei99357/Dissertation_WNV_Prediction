from pathlib import Path
from sklearn import ensemble, metrics
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp
import seaborn as sns

# ======= Set correct paths =======
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[3] / "data_demo" / "CA_human_data_2004_to_2023_final_all_counties_CDPH_scraped.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ======= Load the dataset =======
data = pd.read_csv(DATA_PATH, index_col=False, header=0)

# Drop columns not needed
data = data.drop([
    "Date", "County", "Latitude", "Longitude",
    "Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count"
], axis=1, errors="ignore")

# Drop empty or constant columns
data = data.dropna(axis=1, how='all')
data = data.loc[:, data.var() != 0]
data = data.reset_index(drop=True)

# Print 0 variance columns
print(data.columns[data.var() == 0])

# Impute Human_Disease_Count
data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)

# ======= Main Loop =======
years = data["Year"].unique()
years.sort()

tuning_years_list = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                     2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

tuning_year_dict = {}
best_hyperparameters_df = pd.DataFrame(columns=[
    "tuning_year", "n_estimators", "max_depth", "min_samples_split",
    "min_samples_leaf", "max_features", "max_samples",
    "max_leaf_nodes", "min_impurity_decrease"
])

max_features_map = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'sqrt', 'log2']

for tuning_year in tuning_years_list:
    print("tuning year:", tuning_year)
    best_hyperparameters = []

    for year in years:
        if year != tuning_year:
            continue

        train = data[data['Year'] < year].copy()
        test = data[data['Year'] == year].copy()

        train = train.dropna().reset_index(drop=True)
        test = test.dropna().reset_index(drop=True)

        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train = train.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")
        test = test.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")

        def objective(params):
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            params['min_samples_split'] = int(params['min_samples_split'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
            params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
            params['max_features'] = max_features_map[params['max_features']]

            rf = ensemble.RandomForestRegressor(**params)
            rf.fit(train, train_labels)
            y_pred = rf.predict(test)
            return -metrics.r2_score(test_labels, y_pred)

        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
            'max_depth': hp.quniform('max_depth', 1, 20, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
            'max_features': hp.choice('max_features', list(range(len(max_features_map)))),
            'max_samples': hp.uniform('max_samples', 0.1, 1),
            'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 1),
            'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.1)
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        best['min_samples_split'] = int(best['min_samples_split'])
        best['min_samples_leaf'] = int(best['min_samples_leaf'])
        best['max_leaf_nodes'] = int(best['max_leaf_nodes'])
        best['max_features'] = max_features_map[best['max_features']]

        best_hyperparameters.append(best)

        best_hyperparameters_df = best_hyperparameters_df.append({
            "tuning_year": year,
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "min_samples_split": best["min_samples_split"],
            "min_samples_leaf": best["min_samples_leaf"],
            "max_features": best["max_features"],
            "max_samples": best["max_samples"],
            "max_leaf_nodes": best["max_leaf_nodes"],
            "min_impurity_decrease": best["min_impurity_decrease"]
        }, ignore_index=True)

    print("best_hyperparameters:", best_hyperparameters)

    rmse_list = []
    r2_list = []

    predict_year = 2005
    for year in years:
        if year < predict_year:
            continue

        train = data[data['Year'] < year].copy()
        test = data[data['Year'] == year].copy()

        train = train.dropna().reset_index(drop=True)
        test = test.dropna().reset_index(drop=True)

        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train = train.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")
        test = test.drop(["Month", "FIPS", "Year"], axis=1, errors="ignore")

        rf = ensemble.RandomForestRegressor(**best_hyperparameters[0])
        rf.fit(train, train_labels)
        y_pred = rf.predict(test)

        mse = metrics.mean_squared_error(test_labels, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(test_labels, y_pred)

        rmse_list.append(rmse)
        r2_list.append(r2)

        predict_year += 1

        print("predict year:", year, ", Q2:", r2)

    tuning_year_dict[tuning_year] = {"q2": r2_list, "rmse": rmse_list}

# Save results
best_hyperparameters_df.to_csv(RESULTS_DIR / "hyperparameter_tuning_rf_impute_0.csv", index=False)

x_axis_years = list(range(2005, 2024))

# ======= Heatmap Q2 =======
fig, ax = plt.subplots(figsize=(30, 25))
q2_array = []
for tuning_year in tuning_years_list:
    q2_array.append(tuning_year_dict[tuning_year]["q2"])

q2_array = np.clip(q2_array, -1, 1)
mask = np.tril(np.ones(q2_array.shape), k=0).astype(bool)
q2_array[mask] = np.nan

sns.heatmap(q2_array, cmap='RdBu', annot=False, linewidths=0.5, ax=ax, center=0, vmax=1, vmin=-1)

cbar = ax.collections[0].colorbar
cbar.set_label("Q2", fontsize=30)
cbar.ax.tick_params(labelsize=25)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')

ax.set_xlabel("Predicting Year", fontsize=30)
ax.set_ylabel("RF Model Using Best Hyperparameter for Predicting Year ____", fontsize=30)

ax.set_xticks(np.arange(len(x_axis_years)) + 0.5)
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5)
ax.set_xticklabels(x_axis_years, fontsize=25, rotation=45)
ax.set_yticklabels(tuning_years_list, fontsize=25)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "q2_tuning_year_heatmap_rf.png", dpi=300)
plt.close()

# ======= Heatmap RMSE =======
fig, ax = plt.subplots(figsize=(30, 25))
rmse_array = []
for tuning_year in tuning_years_list:
    rmse_array.append(tuning_year_dict[tuning_year]["rmse"])

sns.heatmap(rmse_array, cmap='Reds', annot=False, linewidths=0.5, ax=ax)

ax.set_xlabel("Predicting Year", fontsize=22)
ax.set_ylabel("RF Model Using Best Hyperparameter for Predicting Year ____", fontsize=22)
plt.title("RMSE Heatmap for RF Hyperparameter Tuning", fontsize=25)

ax.set_xticks(np.arange(len(x_axis_years)) + 0.5)
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5)
ax.set_xticklabels(x_axis_years, fontsize=18)
ax.set_yticklabels(tuning_years_list, fontsize=18)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "rmse_tuning_year_heatmap_rf.png", dpi=300)
plt.close()