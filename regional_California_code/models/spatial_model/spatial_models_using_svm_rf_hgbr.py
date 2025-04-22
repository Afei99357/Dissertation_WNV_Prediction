from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn import ensemble, metrics
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from hyperopt import fmin, tpe, hp

# Set paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2] / "data_demo" / "CA_human_data_2004_to_2023_final_all_counties_CDPH_scraped.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SHAP_DIR = RESULTS_DIR / "local_shap"
LOCAL_SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Load and preprocess data
data = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
data = data.drop(columns=["Latitude", "Longitude", "Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count"], errors='ignore')
data = data.dropna(axis=1, how='all').reset_index(drop=True)

# Fill missing values
data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)
data = data.dropna().reset_index(drop=True)

# Rename columns
rename_map = {
    "ONI": "Oceanic Niño Index",
    'Evergreen/Deciduous Needleleaf Trees': 'Needleleaf Trees',
    'Evergreen Broadleaf Trees': 'Broadleaf Trees',
    'u10_1m_shift': 'Eastward wind',
    'v10_1m_shift': 'Northward wind',
    't2m_1m_shift': 'Temperature',
    'lai_lv_1m_shift': 'Low vegetation',
    'sf_1m_shift': 'Snowfall',
    'sro_1m_shift': 'Surface runoff',
    'tp_1m_shift': 'Total precipitation'
}
data = data.rename(columns=rename_map)

# Drop columns with zero variance
excluded_cols = ["Date", "County"]
data = data.drop(columns=[col for col in data.columns if col not in excluded_cols and data[col].var() == 0])

# Train/test split
train = data[data['Year'] < 2019].copy()
test = data[data['Year'] >= 2019].copy()

# Prepare training and testing sets
train_hyper_tune = train.sample(frac=0.8, random_state=0)
train_validation = train.drop(train_hyper_tune.index)

train_hyper_tune = train_hyper_tune.sort_values(by=['County', 'Year', 'Month'])
train_validation = train_validation.sort_values(by=['County', 'Year', 'Month'])
test = test.sort_values(by=['County', 'Year', 'Month'])

# Extract labels
y_train_hyper = train_hyper_tune.pop("Human_Disease_Count").values
y_train_val = train_validation.pop("Human_Disease_Count").values
y_test = test.pop("Human_Disease_Count").values

# Drop non-feature columns
drop_cols = ["Month", "FIPS", "Year", "Date", "County"]
test_info = test[["Year", "Month", "FIPS"]]
train_hyper_tune = train_hyper_tune.drop(columns=drop_cols)
train_validation = train_validation.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# Standardize features
scaler = StandardScaler()
train_hyper_tune = pd.DataFrame(scaler.fit_transform(train_hyper_tune), columns=train_hyper_tune.columns)
train_validation = pd.DataFrame(scaler.transform(train_validation), columns=train_validation.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

####################################
# SVM Hyperparameter Tuning
####################################
kernel_map = ["rbf", "linear", "poly", "sigmoid"]
gamma_map = ["scale", "auto"]

svm_space = {
    "C": hp.uniform("C", 0.1, 10),
    "epsilon": hp.uniform("epsilon", 0.1, 5),
    "kernel": hp.choice("kernel", kernel_map),
    "gamma": hp.choice("gamma", gamma_map)
}

def svm_objective(params):
    model = SVR(**params)
    model.fit(train_hyper_tune, y_train_hyper)
    preds = model.predict(train_validation)
    return -metrics.r2_score(y_train_val, preds)

best_svm = fmin(fn=svm_objective, space=svm_space, algo=tpe.suggest, max_evals=200)
best_svm['kernel'] = kernel_map[int(best_svm['kernel'])]
best_svm['gamma'] = gamma_map[int(best_svm['gamma'])]

svm_model = SVR(**best_svm)
svm_model.fit(train_hyper_tune, y_train_hyper)
svm_preds = np.clip(svm_model.predict(test), 0, None)

####################################
# Random Forest Hyperparameter Tuning
####################################
max_features_map = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'sqrt', 'log2']

rf_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.choice('max_features', list(range(len(max_features_map)))),
    'max_samples': hp.uniform('max_samples', 0.1, 1),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 1),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.1)
}

def rf_objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
    params['max_features'] = max_features_map[params['max_features']]
    model = ensemble.RandomForestRegressor(**params)
    model.fit(train_hyper_tune, y_train_hyper)
    preds = model.predict(train_validation)
    return -metrics.r2_score(y_train_val, preds)

best_rf = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=100)
best_rf['max_features'] = max_features_map[best_rf['max_features']]
for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']:
    best_rf[key] = int(best_rf[key])

rf_model = ensemble.RandomForestRegressor(**best_rf)
rf_model.fit(train_hyper_tune, y_train_hyper)
rf_preds = rf_model.predict(test)

####################################
# Histogram-Based Gradient Boosting
####################################
scoring_map = ['loss', 'neg_mean_squared_error', 'neg_mean_absolute_error']

hgbr_space = {
    'max_depth': hp.quniform('max_depth', 1, 30, 1),
    'max_iter': hp.quniform('max_iter', 100, 1000, 100),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'l2_regularization': hp.uniform('l2_regularization', 0.0, 1.0),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 10),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_bins': hp.quniform('max_bins', 10, 255, 5),
    'scoring': hp.choice('scoring', scoring_map)
}

def hgbr_objective(params):
    for key in ['max_depth', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'max_bins']:
        params[key] = int(params[key])
    model = ensemble.HistGradientBoostingRegressor(**params)
    model.fit(train_hyper_tune, y_train_hyper)
    preds = model.predict(train_validation)
    return -metrics.r2_score(y_train_val, preds)

best_hgbr = fmin(fn=hgbr_objective, space=hgbr_space, algo=tpe.suggest, max_evals=100)
best_hgbr['scoring'] = scoring_map[int(best_hgbr['scoring'])]
for key in ['max_depth', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'max_bins']:
    best_hgbr[key] = int(best_hgbr[key])

hgbr_model = ensemble.HistGradientBoostingRegressor(**best_hgbr)
hgbr_model.fit(train_hyper_tune, y_train_hyper)
hgbr_preds = hgbr_model.predict(test)

####################################
# Results Compilation and Saving
####################################

results = pd.DataFrame({
    "Model": ["SVM", "RF", "HGBR"],
    "Q2": [metrics.r2_score(y_test, svm_preds), metrics.r2_score(y_test, rf_preds), metrics.r2_score(y_test, hgbr_preds)],
    "MSE": [metrics.mean_squared_error(y_test, svm_preds), metrics.mean_squared_error(y_test, rf_preds), metrics.mean_squared_error(y_test, hgbr_preds)]
})

results.to_csv(RESULTS_DIR / "prediction_hyperparameter_tuning_svm_rf_hgbr.csv", index=False)

# SHAP for SVM
explainer = shap.Explainer(svm_model.predict, test)
shap_values = explainer(test)

shap_df = pd.DataFrame(shap_values.values, columns=test.columns)
shap_df = pd.concat([test_info.reset_index(drop=True), shap_df], axis=1)
shap_df.to_csv(RESULTS_DIR / "svm_global_shap_values.csv", index=False)

plt.figure(figsize=(30, 10))
shap.plots.bar(shap_values, show=False, max_display=18)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "svm_global_shap_plot.png")
plt.close()

for idx in range(len(test)):
    plt.figure(figsize=(60, 20))
    shap.plots.bar(shap_values[idx], show=False, max_display=17)
    plt.tight_layout()
    plt.savefig(LOCAL_SHAP_DIR / f"svm_local_shap_plot_{test_info.iloc[idx]['Year']}_{test_info.iloc[idx]['Month']}_{test_info.iloc[idx]['FIPS']}.png")
    plt.close()
