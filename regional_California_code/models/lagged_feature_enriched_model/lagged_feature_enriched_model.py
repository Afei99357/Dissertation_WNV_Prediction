from sklearn.svm import SVR
from sklearn import ensemble, metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp
import shap
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

from stack_data import markers_from_ranges
from torch.optim.rprop import Rprop


def bootstrapping(test_df:pd.DataFrame,
                  test_labels:pd.Series,
                  model,
                  model_name:str,
                  output_path:str,
                  scalar,
                  n_iterations:int,
                  ):
    ###################################################
    # bootstrapping to train the svm model and predict the testing data, get the q2 and mse
    ####################################################
    # Arrays to store bootstrap results
    bootstrap_r2 = []
    bootstrap_mse = []

    # Perform bootstrapping
    for _ in range(n_iterations):
        print(f"Iteration: {_}")
        # Resample test set with replacement
        test_resample, labels_resample = resample(test_df, test_labels)

        # Predict using the trained model
        predictions_resample = model.predict(test_resample)

        # Inverse the scaling
        if scalar is not None:
            predictions_resample = scalar.inverse_transform(predictions_resample.reshape(-1, 1)).reshape(-1)

        # Calculate metrics
        r2_resample = metrics.r2_score(labels_resample, predictions_resample)
        mse_resample = metrics.mean_squared_error(labels_resample, predictions_resample)

        # Store results
        bootstrap_r2.append(r2_resample)
        bootstrap_mse.append(mse_resample)

    ## calculate confidence interval for r2 and mse
    r2_mean = round(np.mean(bootstrap_r2), 2)
    mse_mean = round(np.mean(bootstrap_mse), 2)

    ## define the confidence interval
    def confidence_interval(data, alpha=0.05):
        lower_bound = np.percentile(data, 100 * alpha / 2, axis=0)
        upper_bound = np.percentile(data, 100 * (1 - alpha / 2), axis=0)
        return round(lower_bound, 2), round(upper_bound, 2)

    r2_lower, r2_upper = confidence_interval(bootstrap_r2)

    mse_lower, mse_upper = confidence_interval(bootstrap_mse)

    ## plot the r2 and mse separately with confidence interval
    plt.figure(figsize=(10, 5))
    plt.hist(bootstrap_r2, bins=30, color='blue', alpha=0.5)
    plt.axvline(r2_mean, color='red', linestyle='dashed', linewidth=2, label=f'mean R2: {r2_mean}')
    plt.axvline(r2_lower, color='purple', linestyle='dashed', linewidth=2,
                label=f'95% confidence interval lower bound: {r2_lower}')
    plt.axvline(r2_upper, color='black', linestyle='dashed', linewidth=2,
                label=f'95% confidence interval upper bound: {r2_upper}')

    ## add legend
    plt.legend(loc='upper left')

    plt.title("Test R2 distribution")
    plt.savefig(
        output_path / f"bootstrapping_r2_distribution_with_{model_name}.png")
    #close the plot
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(bootstrap_mse, bins=30, color='blue', alpha=0.5)
    plt.axvline(mse_mean, color='red', linestyle='dashed', linewidth=2, label=f'mean MSE: {mse_mean}')
    plt.axvline(mse_lower, color='purple', linestyle='dashed', linewidth=2,
                label=f'95% confidence interval lower bound: {mse_lower}')
    plt.axvline(mse_upper, color='black', linestyle='dashed', linewidth=2,
                label=f'95% confidence interval upper bound: {mse_upper}')

    ## add legend
    plt.legend(loc='upper right')

    plt.title("MSE distribution")
    plt.savefig(
        output_path / f"bootstrapping_mse_with_{model_name}.png")
    #close the plot
    plt.close()

def calculate_shap_values(test: pd.DataFrame, test_info: pd.DataFrame, model, test_columns_shap: list,
                          output_path: str, method:str):
    ## shap values for SVM
    def f(X):
        return model.predict(X)

    ## shap values
    explainer = shap.Explainer(f, test)(test)

    ## plot global shap values
    ## plot global bar plot where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
    plt.figure(figsize=(30, 10))

    ## make a directory if not exist
    if not (output_path / f"{method}_shap_values").exists():
        (output_path / f"{method}_shap_values").mkdir(parents=True, exist_ok=True)

    shap.plots.bar(explainer, show=False, max_display=18)
    plt.tight_layout()
    plt.savefig(
        output_path / f"{method}_shap_values/{method}_global_shap_plot_predict_19_to_23_with_daylight.png")
    plt.close()

    ##output sharp values to a csv file
    shap_values = explainer.values
    shap_values_df = pd.DataFrame(shap_values, columns=test_columns_shap)
    ## add the year, month and county information to the shap values
    shap_values_df["Year"] = test_info["Year"].values
    shap_values_df["Month"] = test_info["Month"].values
    shap_values_df["FIPS"] = test_info["FIPS"].values

    ## save the shap values to a csv file
    shap_values_df.to_csv(
        output_path / f"{method}_shap_values/{method}_shap_values_predict_19_to_23_with_daylight.csv",
        index=False)
    #######

    ## make a directory if not exist
    local_shap_dir = output_path / f"{method}_shap_values" / "local_shap"
    if not local_shap_dir.exists():
        local_shap_dir.mkdir(parents=True, exist_ok=True)

    ### plot local shap values, individual shap values
    ## for each sample in the x_test, get the year, month and county information for output file
    for i in range(len(shap_values_df)):
        plt.figure(figsize=(60, 20))

        ## adding padding to the plot
        plt.subplots_adjust(left=0.4, right=0.6, top=0.9, bottom=0.1)

        shap.plots.bar(explainer[i], show=False, max_display=17)
        month = test_info["Month"].values[i]
        year = test_info["Year"].values[i]
        county = test_info["FIPS"].values[i]
        plt.tight_layout()
        plt.savefig(
            output_path / f"{method}_shap_values/local_shap/svm_local_shap_plot_{year}_{month}_{county}.png")
        plt.close()


def plot_prediction_results(results:pd.DataFrame, output_path:str):
    ## plot the R2 and MSE scores in a line charts, with 2 y axis, one for R2 and one for MSE
    # Extract data for plotting
    models = ['SVM', 'RF', 'HGBR']
    r2_values = results["R2"].tolist()  # Q^2 values for the models
    mse_values = results["MSE"].tolist()

    ## plot the R2 and MSE with the lag time with y1 axis as Q2 and y2 axis as MSE
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R2', color=color)
    # ax1.plot(range(0, 3), r2_values, color=color)
    ax1.scatter(range(0, 3), r2_values, color=color, marker='o',s=100, label='R2')
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('MSE', color=color)
    # ax2.plot(range(0, 3), mse_values, color=color)
    ax2.scatter(range(0, 3), mse_values, color=color, marker='v',s=100, label='MSE')
    ax2.tick_params(axis='y', labelcolor=color)

    ## legend
    fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5))

    ## x axis ticks label
    plt.xticks(range(0, 3), models)

    fig.tight_layout()

    plt.savefig(
        output_path / "R2_MSE_with_models.png",
        dpi=300)
    plt.close()


def run_svm_model(
                    tuning_df:pd.DataFrame,
                    validation_df:pd.DataFrame,
                    test_df:pd.DataFrame,
                    train_hyper_labels:pd.Series,
                    train_validation_labels:pd.Series,
                    test_info:pd.DataFrame,
                    output_path:str):
    ## standardize the labels
    # output_distribution = 'normal'
    # scalar_label = QuantileTransformer(output_distribution=output_distribution)
    scalar_label = StandardScaler()
    train_hyper_labels_transform = scalar_label.fit_transform(train_hyper_labels.reshape(-1, 1)).reshape(-1)
    train_validation_labels_transform = scalar_label.transform(train_validation_labels.reshape(-1, 1)).reshape(-1)

    shap_value_columns = test_df.columns

    ## scale the data
    scaler = StandardScaler()
    # scaler = QuantileTransformer(output_distribution=output_distribution)
    tuning_df = pd.DataFrame(scaler.fit_transform(tuning_df), columns=tuning_df.columns)
    validation_df = pd.DataFrame(scaler.transform(validation_df), columns=validation_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    ##### SVM hyperparameter tuning and prediction#####
    # Define the mappings for kernel and gamma
    kernel_map = ["rbf", "linear", "poly", "sigmoid"]
    gamma_map = ["scale", "auto"]

    ## tuning the hyperparameters using hyperopt
    def hyper_tune_svm(params):
        svm = SVR(**params)
        svm.fit(tuning_df, train_hyper_labels_transform)
        y_predict = svm.predict(validation_df)
        ## CALCULATE r2
        r2 = metrics.r2_score(train_validation_labels_transform, y_predict)
        return -r2

    space = {
        "C": hp.uniform("C", 0.1, 10),
        "epsilon": hp.uniform("epsilon", 0.1, 5),
        "kernel": hp.choice("kernel", ["rbf", "linear", "poly", "sigmoid"]),
        "gamma": hp.choice("gamma", ["scale", "auto"])
    }

    best_svm = fmin(fn=hyper_tune_svm, space=space, algo=tpe.suggest, max_evals=100)

    # Map the best indices back to the corresponding string values
    best_svm['kernel'] = kernel_map[int(best_svm['kernel'])]
    best_svm['gamma'] = gamma_map[int(best_svm['gamma'])]

    ##print the best hyperparameters
    print("Best hyperparameters for SVM: ", best_svm)

    ## save the model with the best hyperparameters
    svm = SVR(**best_svm)
    svm.fit(tuning_df, train_hyper_labels_transform)

    ## Predict the test data
    predictions_svm = svm.predict(test_df)

    ## inverse the scaling
    predictions_svm = scalar_label.inverse_transform(predictions_svm.reshape(-1, 1)).reshape(-1)

    ## calculate the shap values
    calculate_shap_values(test_df, test_info, svm , shap_value_columns, output_path, "svm")

    return {'prediction_result': predictions_svm, 'best_hyperparameters': best_svm, 'model': svm, 'scalar': scalar_label, 'standardized_test_df': test_df}

def run_rf_model(
                    train_hyper_tune:pd.DataFrame,
                    train_validation:pd.DataFrame,
                    test:pd.DataFrame,
                    train_hyper_labels:pd.Series,
                    train_validation_labels:pd.Series,
                    test_info:pd.DataFrame,
                    output_path:str
):
    # Define the mappings
    max_features_map = [  # Integer
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,  # Floats in range (0.0, 1.0]
        'sqrt', 'log2'  # Strings
    ]

    shap_value_columns = test.columns

    ## tuning the hyperparameters using hyperopt
    def hyper_tune_rf(params):
        # int the hyperparameters
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        params['max_leaf_nodes'] = int(params['max_leaf_nodes'])

        # map max_features correctly
        params['max_features'] = max_features_map[params['max_features']]

        # random forest
        rf = ensemble.RandomForestRegressor(**params)
        rf.fit(train_hyper_tune, train_hyper_labels)
        y_predict = rf.predict(train_validation)
        ## CALCULATE r2
        r2 = metrics.r2_score(train_validation_labels, y_predict)
        return -r2

    # Define the hyperparameter space for random forest
    space = {'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
             'max_depth': hp.quniform('max_depth', 1, 20, 1),
             'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
             'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
             'max_features': hp.choice('max_features', list(range(len(max_features_map)))),
             'max_samples': hp.uniform('max_samples', 0.1, 1),
             'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 1),
             'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.1)
             }

    best_rf = fmin(fn=hyper_tune_rf, space=space, algo=tpe.suggest, max_evals=100)

    # int the hyperparameters
    best_rf['n_estimators'] = int(best_rf['n_estimators'])
    best_rf['max_depth'] = int(best_rf['max_depth'])
    best_rf['min_samples_split'] = int(best_rf['min_samples_split'])
    best_rf['min_samples_leaf'] = int(best_rf['min_samples_leaf'])
    best_rf['max_leaf_nodes'] = int(best_rf['max_leaf_nodes'])
    # Map max_features index to actual value
    best_rf['max_features'] = max_features_map[best_rf['max_features']]

    ##print the best hyperparameters
    print("Best hyperparameters for RF: ", best_rf)

    ## save the model with the best hyperparameters
    rf = ensemble.RandomForestRegressor(**best_rf)
    rf.fit(train_hyper_tune, train_hyper_labels)

    ## Predict the test data
    predictions_rf = rf.predict(test)

    ## calculate the shap values
    calculate_shap_values(test, test_info, rf, shap_value_columns, output_path, "rf")

    return {'prediction_result': predictions_rf, 'best_hyperparameters': best_rf, 'model': rf, 'scalar': None}

def run_hgbr_model(
                    train_hyper_tune:pd.DataFrame,
                    train_validation:pd.DataFrame,
                    test:pd.DataFrame,
                    train_hyper_labels:pd.Series,
                    train_validation_labels:pd.Series,
                    test_info:pd.DataFrame,
                    output_path:str
):
    ## tuning the hyperparameters using hyperopt
    # Define the mappings for kernel and gamma
    scoring_map = ['loss', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    def hyper_tune_hgbr(params):
        # int the hyperparameters
        params['max_depth'] = int(params['max_depth'])
        params['max_iter'] = int(params['max_iter'])
        params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        params['max_bins'] = int(params['max_bins'])

        # HistGradientBoostingRegressor
        HGBR = ensemble.HistGradientBoostingRegressor(**params)
        HGBR.fit(train_hyper_tune, train_hyper_labels)
        y_predict = HGBR.predict(train_validation)
        ## CALCULATE r2
        r2 = metrics.r2_score(train_validation_labels, y_predict)
        return -r2

    shap_value_columns = test.columns

    # Define the hyperparameter space for hgbr
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

    best_hgbr = fmin(fn=hyper_tune_hgbr, space=space, algo=tpe.suggest, max_evals=100)

    # Map the best indices back to the corresponding string values
    best_hgbr['scoring'] = scoring_map[int(best_hgbr['scoring'])]

    ## int the best hyperparameters
    best_hgbr["max_depth"] = int(best_hgbr["max_depth"])
    best_hgbr["max_iter"] = int(best_hgbr["max_iter"])
    best_hgbr["max_leaf_nodes"] = int(best_hgbr["max_leaf_nodes"])
    best_hgbr["min_samples_leaf"] = int(best_hgbr["min_samples_leaf"])
    best_hgbr["max_bins"] = int(best_hgbr["max_bins"])

    ##print the best hyperparameters
    print("Best hyperparameters for HGBR: ", best_hgbr)

    ## save the model with the best hyperparameters
    hgbr = ensemble.HistGradientBoostingRegressor(**best_hgbr)
    hgbr.fit(train_hyper_tune, train_hyper_labels)

    ## Predict the test data
    predictions_hgbr = hgbr.predict(test)

    ## calculate the shap values
    calculate_shap_values(test_df, test_info, hgbr, shap_value_columns, output_path, "hgbr")

    return {'prediction_result': predictions_hgbr, 'best_hyperparameters': best_hgbr, 'model': hgbr, 'scalar': None}

def calculate_metrics(predictions:np.array, test_labels:np.array):
    ## get the R2 score
    r2 = metrics.r2_score(test_labels, predictions)

    ## get the mse score
    mse = metrics.mean_squared_error(test_labels, predictions)

    ## print the R2 score
    print("test R2: ", r2)
    ## print the mse score
    print("MSE: ", mse)

    return {"R2": r2, "MSE": mse}

def data_preprocessing(data:pd.DataFrame, output_path:str):
    # Drop columns that are not features and drop target
    data = data.drop([
        # "Date",
        # "County",
        "Latitude",
        "Longitude",
        "Total_Bird_WNV_Count",
        "Mos_WNV_Count",
        "Horse_WNV_Count",
        # "lai_hv_1m_shift"
    ], axis=1)

    # Drop columns if all the values in the columns are the same or all nan
    data = data.dropna(axis=1, how='all')

    # Reindex the data
    data = data.reset_index(drop=True)

    # Get the unique years and sort them
    years = data["Year"].unique()
    years.sort()

    ## fill missing values in Human_Disease_Count with 0
    data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)

    # ## drop rows with missing values
    data = data.dropna().reset_index(drop=True)

    ## RENAME COLUMN
    data = data.rename(columns={"ONI": "Oceanic Niño Index", 'Evergreen/Deciduous Needleleaf Trees': 'Needleleaf Trees',
                                'Evergreen Broadleaf Trees': 'Broadleaf Trees', 'u10_1m_shift': 'Eastward wind',
                                'v10_1m_shift': 'Northward wind', 't2m_1m_shift': 'Temperature',
                                'lai_hv_1m_shift': 'High vegetation', 'lai_lv_1m_shift': 'Low vegetation',
                                'sf_1m_shift': 'Snowfall', 'sro_1m_shift': 'Surface runoff',
                                'tp_1m_shift': 'Total precipitation'})

    ## for each column, if the whole column have zero variance, drop the column
    excluded_cols = ["Date", "County"]

    # Drop columns with zero variance except for 'Date' and 'County'
    data = data.drop(columns=[col for col in data.columns if col not in excluded_cols and data[col].var() == 0])

    # 80% of the train data is used for hyperparameter tuning, 20% for validation, and save the model
    train_hyper_tune = data[data['Year'] < 2016].copy()
    train_validation = data[(data['Year'] >= 2016) & (data['Year'] < 2019)].copy()
    test = data[data['Year'] >= 2019].copy()

    ## calculate the differencing between the current month and the previous month
    train_hyper_tune = train_hyper_tune.sort_values(by=['County', 'Year', 'Month'])
    train_hyper_tune['diff'] = train_hyper_tune.groupby('County')['Human_Disease_Count'].diff(1)
    train_validation = train_validation.sort_values(by=['County', 'Year', 'Month'])
    train_validation['diff'] = train_validation.groupby('County')['Human_Disease_Count'].diff(1)
    test = test.sort_values(by=['County', 'Year', 'Month'])
    test['diff'] = test.groupby('County')['Human_Disease_Count'].diff(1)

    lag_time = 3

    ## create lagged values for the linear model
    for i in range(1, lag_time + 1):
        train_hyper_tune['t-' + str(i)] = train_hyper_tune.groupby('County')['diff'].shift(i)
        train_validation['t-' + str(i)] = train_validation.groupby('County')['diff'].shift(i)
        test['t-' + str(i)] = test.groupby('County')['diff'].shift(i)

    ## remove the rows with nan values in diff, t-1, t-2, t-3
    lag_list = [f't-{x}' for x in range(1, lag_time + 1)] + ['diff']

    tuning_df = train_hyper_tune.dropna(subset=lag_list)
    validation_df = train_validation.dropna(subset=lag_list)
    test_df = test.dropna(subset=lag_list)

    ## get labels
    train_hyper_labels = tuning_df.pop("Human_Disease_Count").values
    train_validation_labels = validation_df.pop("Human_Disease_Count").values
    test_labels = test_df.pop("Human_Disease_Count").values

    ## drop unnecessary columns
    tuning_df = tuning_df.drop(["Month", "FIPS", "Year"], axis=1)
    validation_df = validation_df.drop(["Month", "FIPS", "Year"], axis=1)
    test_year_list = test_df.pop("Year").values
    test_month_list = test_df.pop("Month").values
    test_FIPS_list = test_df.pop("FIPS").values

    # create a new dataframe to store the lagged values
    test_lag_info = pd.DataFrame()
    test_lag_info['Year'] = test_year_list
    test_lag_info['Month'] = test_month_list
    test_lag_info['FIPS'] = test_FIPS_list
    ## get the t1, t2, t3, t4 columns in test in to a lists
    for i in range(1, lag_time + 1):
        test_lag_info[f't-{i}'] = test_df[f't-{i}'].values

    ## drop the date column, county column and the diff column
    tuning_df = tuning_df.drop(["Date", "County", 'diff'], axis=1)
    validation_df = validation_df.drop(["Date", "County", 'diff'], axis=1)
    test_df = test_df.drop(["Date", "County", "diff"], axis=1)

    return tuning_df, validation_df, test_df, train_hyper_labels, train_validation_labels, test_labels, test_lag_info, lag_time

if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = Path(__file__).resolve().parents[
                    2] / "data_demo" / "CA_human_data_2004_to_2023_final_all_counties_CDPH_scraped.csv"
    OUTPUT_PATH = BASE_DIR / "results"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH, index_col=False, header=0)
    output_path = OUTPUT_PATH

    #
    R2_list = []
    mse_list = []
    best_hyperparameters = []

    ## preprocess the data
    tuning_df, validation_df, test_df, train_hyper_labels, train_validation_labels, test_labels, test_info, lag_time = data_preprocessing(data, output_path)

    # run the svm model
    svm_result = run_svm_model(
        tuning_df.copy(),
        validation_df.copy(),
        test_df.copy(),
        train_hyper_labels,
        train_validation_labels,
        test_info,
        output_path)

    ## run the bootstrapping
    bootstrapping(svm_result['standardized_test_df'], test_labels, svm_result['model'], 'SVM', output_path, svm_result['scalar'], 1000)

    ## calculate the metrics
    metrics_results_svm = calculate_metrics(svm_result['prediction_result'], test_labels)

    R2_list.append(metrics_results_svm["R2"])
    mse_list.append(metrics_results_svm["MSE"])
    best_hyperparameters.append(svm_result['best_hyperparameters'])


    ### run the random forest model
    rf_result = run_rf_model(
        tuning_df.copy(),
        validation_df.copy(),
        test_df.copy(),
        train_hyper_labels,
        train_validation_labels,
        test_info,
        output_path)

    ## run the bootstrapping
    bootstrapping(test_df, test_labels, rf_result['model'], 'RF', output_path, rf_result['scalar'], 1000)

    ## calculate the metrics
    metrics_result_rf = calculate_metrics(rf_result['prediction_result'], test_labels)

    R2_list.append(metrics_result_rf["R2"])
    mse_list.append(metrics_result_rf["MSE"])
    best_hyperparameters.append(rf_result['best_hyperparameters'])

    ## run the hgbr model
    hgbr_result = run_hgbr_model(
        tuning_df.copy(),
        validation_df.copy(),
        test_df.copy(),
        train_hyper_labels,
        train_validation_labels,
        test_info,
        output_path)

    ## run the bootstrapping
    bootstrapping(test_df, test_labels, hgbr_result['model'], 'HGBR', output_path, hgbr_result['scalar'], 1000)

    ## calculate the metrics
    metrics_result_hgbr = calculate_metrics(hgbr_result['prediction_result'], test_labels)

    R2_list.append(metrics_result_hgbr["R2"])
    mse_list.append(metrics_result_hgbr["MSE"])
    best_hyperparameters.append(hgbr_result['best_hyperparameters'])

    ## create a df to store the results
    results = pd.DataFrame({
        "Model": ['SVM', 'RF', 'HGBR'],
        "R2": R2_list,
        "MSE": mse_list,
        "Best_Hyperparameters": best_hyperparameters
    })

    ## save the results to a csv file
    results.to_csv(
        output_path / "time_series_prediction_hyperparameter_tuning_svm_rf_hgbr.csv",
        index=False)

    # results = pd.read_csv(output_path / "time_series_prediction_hyperparameter_tuning_svm_rf_hgbr_3.csv")

    ## plot the results
    plot_prediction_results(results, output_path)

