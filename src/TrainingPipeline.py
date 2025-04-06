# #############################################################################
# ##### Competition: DATATHON 2025 - ETH Zürich
# ##### Challenge: ALPIQ Challenge (Predict Energy Consumption for Italy and Spain)
# ##### Team: Gradient Descenters
# ##### Members: Kevin Riehl, Alexander Faroux, Cedric Zeiter, Anja Sjöström
# #############################################################################




# #############################################################################
# IMPORTS
# #############################################################################
import pandas as pd
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor



# #############################################################################
# DATA LOADING
# #############################################################################
def load_data(path, country, date_from, date_to, date_format="%Y-%m-%d %H:%M:%S"):
    """
    This function loadas all data into a dataframe.
    This includes consumption data (to be predicted) and features (to explain).
    """
        # LOAD TABLES
    consumptions_path = join(path, "historical_metering_data_" + country + ".csv")
    features_path = join(path, "spv_ec00_forecasts_es_it.xlsx")
    consumptions = pd.read_csv(consumptions_path, index_col=0, parse_dates=True, date_format=date_format)
    features = pd.read_excel(features_path, sheet_name=country, index_col=0, parse_dates=True, date_format=date_format,)
    holidays = pd.read_excel(path+"/holiday_"+country+".xlsx")["holiday_"+country]
    holidays = pd.to_datetime(holidays)
    df_oil, df_msci = load_economic_features()
    features_rollouts = pd.read_csv(path+"/rollout_data_"+country+".csv")
        # MODIFY COLUMNS & FILTER RELEVANT TIME SPAN
    consumptions['total_consumption'] = consumptions.sum(axis=1)
    consumptions = consumptions.reset_index()
    consumptions = consumptions.rename(columns={"DATETIME": "time"})
    consumptions = consumptions[(consumptions['time'] >= date_from) & (consumptions['time'] <= date_to)]
    features = features.reset_index()
    features = features.rename(columns={"index": "time"})
    features = features[(features['time'] >= date_from) & (features['time'] <= date_to)]
    features_rollouts = features_rollouts.rename(columns={"DATETIME": "time"})
    features_rollouts['time'] = pd.to_datetime(features_rollouts['time'])
    features_rollouts = features_rollouts[(features_rollouts['time'] >= date_from) & (features_rollouts['time'] <= date_to)]
        # CREATE FINAL FEATURE DATASET
    features = features.merge(features_rollouts, on="time", how="left")
    features = features.merge(load_time_features(features, holidays, date_from), on="time", how="left")
    features['date'] = pd.to_datetime(features['time']).dt.normalize()
    features = features.merge(df_oil, on="date", how="left")
    features = features.merge(df_msci, on="date", how="left")
    features = features.merge(load_summer_time_features(features), on="time", how="left")
        # DETERMINE CUSTOMERS
    customer_names = []
    for column_name in consumptions.columns:
        if "VALUEMWH" in column_name:
            customer_names.append("_".join(column_name.split("_")[1:]))
    return consumptions, features, customer_names

def load_economic_features():
    """
    This function generates a dataframe with features on economy.
    For each day, following features are generated / loaded:
        - OilOpen (daily opening price of oil, USD)
        - MSCI_IT (daily economic activtiy of Italy, according to MSCI Italy Index)
        - MSCI_ES (daily economic activtiy of Spain, according to MSCI Spain Index)
    """
    df_oil = pd.read_csv(input_data_path2+"oil_prices_open.csv")
    df_oil['date'] = pd.to_datetime(df_oil['Date'])
    df_oil = df_oil[["date", "OilOpen"]]
    df_oil['date'] = pd.to_datetime(df_oil['date']).dt.tz_localize(None)
    df_msci = pd.read_csv(input_data_path3+"MSCI_"+country+".csv", sep=",")
    if country == "ES":
        df_msci = pd.read_csv(input_data_path3+"MSCI_"+country+".csv", sep=";")
        df_msci['date'] = pd.to_datetime(df_msci['Date'])
        df_msci = df_msci[["date", "Open"]]
        df_msci = df_msci.rename(columns={"Open":"MSCI_national"})
    else:
        df_msci = pd.read_csv(input_data_path3+"MSCI_"+country+".csv", sep=",")
        df_msci['date'] = pd.to_datetime(df_msci['Date'])
        df_msci = df_msci[["date", "Close"]]
        df_msci = df_msci.rename(columns={"Close":"MSCI_national"})
    df_msci['date'] = pd.to_datetime(df_msci['date']).dt.tz_localize(None)
    return df_oil, df_msci

def load_time_features(features, holidays, date_from):
    """
    This function generates a dataframe with features on time.
    For each hour slot, following features are generated:
        - day_nr_inc  (increasing number of days since date_from)
        - is_holiday  (whether is holiday or not)
        - is_weekend  (whether is weekend or not)
        - month       (month)
        - week        (week in year)
        - day_of_week (day in week)
        - year        (year)
        - month_X     (whether it is month X)
        - day_X       (whether it is day X)
    Parameters
    ----------
    features : pd:DataFrame
        DataFrame with "time" column to know for which times to poulate summer time features.
    holidays: pd:DataFrame
        DataFrame with information on which holidays existed
    date_From: date:DateTime
        Reference date, from which the dataset should start.
        
    Returns
    -------
    features_time: pd.DataFrame
        DataFrame containing ["time", "dst_change_day", "dst_change_hour", "is_summer_time"].
    """
    features_time = features.copy()
    features_time = features_time[["time"]]
    features_time['day_nr_inc'] = (features_time['time'] - date_from).dt.days
    features_time['is_holiday'] = features_time['time'].dt.date.isin(holidays.dt.date)
    features_time['is_weekend'] = features_time['time'].dt.dayofweek >= 5
    features_time['month'] = features_time['time'].dt.month
    features_time['week'] = features_time['time'].dt.isocalendar().week
    features_time['day_of_week'] = features_time['time'].dt.dayofweek + 1
    features_time['year'] = features_time['time'].dt.year
    features_time['hour'] = features_time['time'].dt.hour
    # Create binary variables
    for day in range(1, 8):
        features_time[f'day_{day}'] = (features_time['time'].dt.dayofweek + 1 == day).astype(int)
    for month in range(1, 13):
        features_time[f'month_{month}'] = (features_time['time'].dt.month == month).astype(int)
    return features_time 

def load_summer_time_features(features):
    """
    This function generates a dataframe with features on summer time.
    For each hour slot, following binary features are generated (1=True, 0=False):
        - dst_change_day  (whether time change took place in EU during that day)
        - dst_change_hour (whether time change took place in EU during that hour)
        - is_summer_time  (whether summertime is active in EU)

    Parameters
    ----------
    features : pd:DataFrame
        DataFrame with "time" column to know for which times to poulate summer time features.

    Returns
    -------
    features_summertime: pd.DataFrame
        DataFrame containing ["time", "dst_change_day", "dst_change_hour", "is_summer_time"].
    """
    dst_periods = {
        2021: {'start': '2021-03-28', 'end': '2021-10-31'},
        2022: {'start': '2022-03-27', 'end': '2022-10-30'},
        2023: {'start': '2023-03-26', 'end': '2023-10-29'},
        2024: {'start': '2024-03-31', 'end': '2024-10-27'},
        2025: {'start': '2025-03-30', 'end': '2025-10-26'}
    }
    def check_dst_change_hour(row):
        year = row['time'].year
        if year in dst_periods:
            dst_start = pd.Timestamp(dst_periods[year]['start'])
            dst_end = pd.Timestamp(dst_periods[year]['end'])
            if row['time'] == dst_start or row['time'] == dst_end:
                return True
        return False    
    def check_dst_change_day(row):
        year = row['time'].year
        if year in dst_periods:
            dst_start = pd.Timestamp(dst_periods[year]['start'])
            dst_end = pd.Timestamp(dst_periods[year]['end'])
            if row['time'] == dst_start or row['time'] == dst_end:
                return True
        return False
    def is_summer_time(row):
        year = row['time'].year
        if year in dst_periods:
            dst_start = pd.Timestamp(dst_periods[year]['start'])
            dst_end = pd.Timestamp(dst_periods[year]['end'])
            if dst_start <= row['time'] <= dst_end:
                return True
        return False
    features_summertime = features.copy()
    features_summertime['dst_change_day'] = features_summertime.apply(check_dst_change_day, axis=1)
    features_summertime['dst_change_hour'] = features_summertime.apply(check_dst_change_hour, axis=1)
    features_summertime['is_summer_time'] = features_summertime.apply(is_summer_time, axis=1)
    features_summertime['dst_change_day'] = features_summertime['dst_change_day'].astype(int)
    features_summertime['dst_change_hour'] = features_summertime['dst_change_hour'].astype(int)
    features_summertime['is_summer_time'] = features_summertime['is_summer_time'].astype(int)
    features_summertime = features_summertime[["time", "dst_change_day", "dst_change_hour", "is_summer_time"]]
    return features_summertime

def get_data_for_customer(customer, consumptions, features):
    Y = consumptions[["time", "VALUEMWHMETERINGDATA_"+customer]]
    Y = Y.rename(columns={"VALUEMWHMETERINGDATA_"+customer:"consumption"})
    X = features[["time", "spv", "temp", "INITIALROLLOUTVALUE_"+customer, "day_nr_inc", "is_holiday", "is_weekend", "month", "week", "day_of_week", "year", "hour", "OilOpen", "MSCI_national", "dst_change_day", "dst_change_hour", "is_summer_time", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7", "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",]]
    return Y, X




# #############################################################################
# PREDICTION MODELS
# #############################################################################
def train_model_MEAN(Y):
    mean = np.mean(Y["consumption"])
    return mean

def do_forecast_MEAN(mean):   
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast["consumption"] = mean
    return Y_forecast 

def train_model_HOURLYMEAN(Y):
    Y['hour'] = Y['time'].dt.hour
    hourly_means = Y.groupby('hour')['consumption'].mean()
    Y_hourly_means = Y.copy()
    Y_hourly_means['true_consumption'] = Y_hourly_means['consumption'] 
    Y_hourly_means['consumption'] = Y['hour'].map(hourly_means)
    return hourly_means

def do_forecast_HOURLYMEAN(hourly_means):
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast["consumption"] = Y_forecast['hour'].map(hourly_means)
    return Y_forecast 

def train_model_HOURLYMEAN7(Y):
    Y['day_of_week'] = Y['time'].dt.dayofweek
    Y['hour'] = Y['time'].dt.hour
    weekly_hourly_means = Y.groupby(['day_of_week', 'hour'])['consumption'].mean()
    Y_hourly_means = Y.copy()
    Y_hourly_means['true_consumption'] = Y_hourly_means['consumption'] 
    Y_hourly_means['consumption'] = Y_hourly_means.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    return weekly_hourly_means

def do_forecast_HOURLYMEAN7(weekly_hourly_means):
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['consumption'] = Y_forecast.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    return Y_forecast 

def train_model_HOURLYMEAN7_GB(X_train, Y_train, customer):
    def getResiduals_HourlyMean7(Y_train):
        weekly_hourly_means = train_model(Y_train, X_train, customer, modelType="hourlymean7")
        Y_fitted = Y_train.copy()
        Y_fitted['day_of_week'] = Y_fitted['time'].dt.dayofweek
        Y_fitted['hour'] = Y_fitted['time'].dt.hour
        Y_fitted['consumption'] = Y_fitted.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
        residuals_train = Y_fitted.copy()
        residuals_train["residual"] = Y_fitted["consumption"] - Y_train["consumption"]
        residuals_train = residuals_train[["time", "residual"]]
        return residuals_train, Y_fitted, weekly_hourly_means
    residuals_train, Y_fitted, weekly_hourly_means = getResiduals_HourlyMean7(Y_train)
    X_train = X_train[["spv", "temp", "day_nr_inc", "is_holiday", "is_weekend", "month", "day_of_week", "hour", "INITIALROLLOUTVALUE_"+customer, "OilOpen", "MSCI_national", "dst_change_day", "dst_change_hour", "is_summer_time"]]
    grad_boost = HistGradientBoostingRegressor().fit(X_train, residuals_train["residual"])
    return [grad_boost, weekly_hourly_means]
    
def do_forecast_HOURLYMEAN7_GB(model, X_test, customer):
    grad_boost = model[0]
    weekly_hourly_means = model[1]
    X_test_sub  = X_test [["spv", "temp", "day_nr_inc", "is_holiday", "is_weekend", "month", "day_of_week", "hour", "INITIALROLLOUTVALUE_"+customer, "OilOpen", "MSCI_national", "dst_change_day", "dst_change_hour", "is_summer_time"]]
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['hourlymean7'] = Y_forecast.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    Y_forecast["GB_residuals"] = grad_boost.predict(X_test_sub)
    Y_forecast["consumption"] = Y_forecast['hourlymean7'] - Y_forecast["GB_residuals"]
    return Y_forecast 

def train_model_GRADBOOST(X,Y):
    X = X.fillna(0)
    Y = Y.fillna(0)
    grad_boost = HistGradientBoostingRegressor().fit(X.drop(['time', 'month'],axis = 1),Y['consumption'])
    return grad_boost

def do_forecast_GRADBOOST(gradboost, X):
    # X = X.fillna(0)
    pred_grad_boost = gradboost.predict(X.drop(['time', 'month'],axis = 1))
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['consumption'] = pred_grad_boost
    return Y_forecast

def train_model_RANDOMFOREST(X,Y):
    X = X.fillna(0)
    Y = Y.fillna(0)
    random_forest = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest (default: 100)
        max_depth=None,    # Maximum depth of the tree (default: None, fully grown trees)
        random_state=42    # Set a random state for reproducibility
    )
    random_forest.fit(X.drop(['time', 'month'], axis=1), Y['consumption'])
    return random_forest

def do_forecast_RANDOMFOREST(random_forest, X):
    X = X.fillna(0)
    pred_random_forest = random_forest.predict(X.drop(['time', 'month'], axis=1))
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['consumption'] = pred_random_forest
    return Y_forecast




def train_model(Y, X, customer, modelType):
    if modelType=="mean":
        model = train_model_MEAN(Y)
    elif modelType=="hourlymean":
        model = train_model_HOURLYMEAN(Y)
    elif modelType=="hourlymean7":
        model = train_model_HOURLYMEAN7(Y)
    elif modelType=="gradboost":
        model = train_model_GRADBOOST(X, Y)
    elif modelType=="hourlymean7_gb":
        model = train_model_HOURLYMEAN7_GB(X, Y, customer)
    elif modelType=="randomforest":
        model = train_model_RANDOMFOREST(X, Y)
    else:
        print("ERR UNKNOWN MODEL ", modelType)
        sys.exit(0)
    return model

def do_forecast(model, modelType, X, customer):
    if modelType=="mean":
        Y_forecast = do_forecast_MEAN(model)
    elif modelType=="hourlymean":
        Y_forecast = do_forecast_HOURLYMEAN(model)
    elif modelType=="hourlymean7":
        Y_forecast = do_forecast_HOURLYMEAN7(model)
    elif modelType=="gradboost":
        Y_forecast = do_forecast_GRADBOOST(model, X)
    elif modelType=="hourlymean7_gb":
        Y_forecast = do_forecast_HOURLYMEAN7_GB(model, X, customer)
    elif modelType=="randomforest":
        Y_forecast = do_forecast_RANDOMFOREST(model, X)
    else:
        print("ERR UNKNOWN MODEL ", modelType)
        sys.exit(0)
    return Y_forecast




# PARAMETERS
input_data_path = "../../../data/1_original/OneDrive_2025-04-05/Alpiq ETHdatathon challenge 2025/datasets2025/"
input_data_path2 = "../../../data/1_original/Alex/"
input_data_path3 = "../../../data/1_original/Kev/"

output_data_path = "../../../data/2_processed/"
countries = ["IT", "ES"]
training_date_from = pd.Timestamp("2022-01-01 00:00:00")
# training_date_to = pd.Timestamp("2022-05-31 23:00:00")
training_date_to = pd.Timestamp("2024-07-31 23:00:00")

training_date_range = [pd.Timestamp("2022-01-01 00:00:00"), pd.Timestamp("2024-05-31 23:00:00")]
testing_date_range = [pd.Timestamp("2024-06-01 00:00:00"), pd.Timestamp("2024-07-31 23:00:00")]


forecast_steps = 24*31
max_threads = 4  # Maximum number of threads to run in parallel
country = "ES"
model_types = ["mean", "hourlymean", "hourlymean7", "hourlymean7_gb", "gradboost", "expsmooth"]
model_types_nonan = ["hourlymean7_gb", "gradboost", "randomforest"]
modelType = "randomforest"












# """
# INFERENCE
fW = open("logger_"+modelType+".txt", "w+")
region_data = {}
for country in ["ES", "IT"]:
    # STEP 1: LOAD DATA
    consumptions, features, customer_names = load_data(input_data_path, country, training_date_from, training_date_to, date_format="%Y-%m-%d %H:%M:%S")
    customer = customer_names[0]
    region_true = []
    region_pred = []
    for customer in customer_names:
        # STEP 2: CLEAN DATA
        Y, X = get_data_for_customer(customer, consumptions, features)
        # STEP 3: TRAIN MODEL
        Y_train = Y.copy()
        Y_test = Y.copy()
        Y_train = Y_train[(Y_train["time"] >= training_date_range[0]) & (Y_train["time"] <= training_date_range[1])]
        Y_test = Y_test[(Y_test["time"] >= testing_date_range[0]) & (Y_test["time"] <= testing_date_range[1])]
        # first_non_nan_index = Y_train['consumption'].first_valid_index()
        # if first_non_nan_index is None:
            # if modelType in model_types_nonan:
                # modelType = "hourlymean7"
        # else:
            # first_non_nan_date = Y_train.loc[first_non_nan_index, 'time']
        # Y_train = Y_train[(Y_train["time"] >= first_non_nan_date)]
        if modelType in model_types_nonan:
            Y_train["consumption"] = Y_train["consumption"].fillna(0)
        Y_train = Y_train.reset_index()
        Y_test = Y_test.reset_index()
        X_train = X.copy()
        X_test = X.copy()
        X_train = X_train[(X_train["time"] >= training_date_range[0]) & (X_train["time"] <= training_date_range[1])]
        # X_train = X_train[(X_train["time"] >= first_non_nan_date)]
        X_test = X_test[(X_test["time"] >= testing_date_range[0]) & (X_test["time"] <= testing_date_range[1])]
        X_train = X_train.reset_index()
        X_test = X_test.reset_index()
        # Train Model
        model = train_model(Y_train, X_train, customer, modelType)
        # STEP 4: PREDICTION
        Y_forecast = do_forecast(model, modelType, X_test, customer)
        # Store Results
        region_true.append(Y_test["consumption"].tolist())
        region_pred.append(Y_forecast["consumption"].tolist())
        # Evaluate For That Company
        eval_score = np.sum(np.abs(np.asarray(Y_forecast["consumption"]) - np.asarray(Y_test["consumption"])))
        print(country, customer, modelType, eval_score)
        fW.write(str(country))
        fW.write("\t")
        fW.write(str(customer))
        fW.write("\t")
        fW.write(str(eval_score))
        fW.write("\t")
        fW.write("\n")
    region_data[country] = {}
    region_data[country]["true"] = region_true
    region_data[country]["pred"] = region_pred
fW.close()



# EVALUATION
team_name = "Gradient Descenters"
absolute_error = {}
portfolio_error = {}
for country in ['ES', 'IT']:
    # Country Error
    country_error = 0
    for company_ctr in range(0, len(region_data[country]["true"])):
        company_true = np.asarray(region_data[country]["true"][company_ctr])
        company_pred = np.asarray(region_data[country]["pred"][company_ctr])
        country_error += np.nansum(np.abs(np.asarray(company_pred - company_true)))
    # Portfolio Country Error
    portfolio_sum_true = []
    portfolio_sum_pred = []
    for company_ctr in range(0, len(region_data[country]["true"])):
        company_true = np.asarray(region_data[country]["true"][company_ctr])
        company_pred = np.asarray(region_data[country]["pred"][company_ctr])
        portfolio_sum_true.append(company_true) 
        portfolio_sum_pred.append(company_pred)
    portfolio_sum_true = np.nansum(np.asarray(portfolio_sum_true), axis=0)
    portfolio_sum_pred = np.nansum(np.asarray(portfolio_sum_pred), axis=0)
    portfolio_country_error = np.sum(np.nansum(np.abs(portfolio_sum_pred-portfolio_sum_true)))
    absolute_error[country] = country_error
    portfolio_error[country] = portfolio_country_error
# Final Forecast Score
forecast_score = (
    1.0*absolute_error['IT'] + 5.0*absolute_error['ES'] + 
    10.0*portfolio_error['IT'] + 50.0*portfolio_error['ES']
)
print(">>", modelType)
print('The team ' + team_name + ' reached a forecast score of ' +
      str(np.round(forecast_score, 0)))




# VISUALIZATION
plt.figure(figsize=(12,6))
ctr = 1
for country in ['ES', 'IT']:
    plt.subplot(1,2,ctr)
    plt.title("Prediction "+country)
    ctr+=1
    # Portfolio Country Error
    portfolio_sum_true = []
    portfolio_sum_pred = []
    for company_ctr in range(0, len(region_data[country]["true"])):
        company_true = np.asarray(region_data[country]["true"][company_ctr])
        company_pred = np.asarray(region_data[country]["pred"][company_ctr])
        portfolio_sum_true.append(company_true) 
        portfolio_sum_pred.append(company_pred)
        
    portfolio_sum_true = np.nansum(np.asarray(portfolio_sum_true), axis=0)
    portfolio_sum_pred = np.nansum(np.asarray(portfolio_sum_pred), axis=0)
    
    plt.plot(portfolio_sum_true, label="true")
    plt.plot(portfolio_sum_pred, label="pred")
    plt.legend()
plt.tight_layout()
# """