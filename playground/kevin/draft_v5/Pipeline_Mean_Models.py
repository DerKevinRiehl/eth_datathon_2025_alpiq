# IMPORTS
import pandas as pd
from os.path import join
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor






# METHODS
def loadData(path, country, date_from, date_to, date_format="%Y-%m-%d %H:%M:%S"):
        # LOAD TABLES
    consumptions_path = join(path, "historical_metering_data_" + country + ".csv")
    features_path = join(path, "spv_ec00_forecasts_es_it.xlsx")
    consumptions = pd.read_csv(consumptions_path, index_col=0, parse_dates=True, date_format=date_format)
    features = pd.read_excel(features_path, sheet_name=country, index_col=0, parse_dates=True, date_format=date_format,)
        # MODIFY COLUMNS
    features = features.reset_index()
    features = features.rename(columns={"index": "time"})
    features = features[(features['time'] >= date_from) & (features['time'] <= date_to)]
    consumptions['total_consumption'] = consumptions.sum(axis=1)
    consumptions = consumptions.reset_index()
    consumptions = consumptions.rename(columns={"DATETIME": "time"})
    consumptions = consumptions[(consumptions['time'] >= date_from) & (consumptions['time'] <= date_to)]
        # LOAD ROLLOUTS
    features2 = pd.read_csv(path+"/rollout_data_"+country+".csv")
    features2 = features2.rename(columns={"DATETIME": "time"})
    features2['time'] = pd.to_datetime(features2['time'])
    features2 = features2[(features2['time'] >= date_from) & (features2['time'] <= date_to)]
        # LOAD HOLIDAYS
    holidays = pd.read_excel(path+"/holiday_"+country+".xlsx")["holiday_"+country]
        # CREATE TIME FEATURES
    features3 = features.copy()
    features3 = features3[["time"]]
    holidays = pd.to_datetime(holidays)
    features3['day_nr_inc'] = (features3['time'] - date_from).dt.days
    features3['is_holiday'] = features3['time'].dt.date.isin(holidays.dt.date)
    features3['is_weekend'] = features3['time'].dt.dayofweek >= 5
    features3['month'] = features3['time'].dt.month
    features3['week'] = features3['time'].dt.isocalendar().week
    features3['day_of_week'] = features3['time'].dt.dayofweek + 1
    features3['year'] = features3['time'].dt.year
    features3['hour'] = features3['time'].dt.hour
        # ECONOMIC DATA
    df_oil = pd.read_csv(input_data_path2+"oil_prices_open.csv")
    df_oil['date'] = pd.to_datetime(df_oil['Date'])
    df_oil = df_oil[["date", "OilOpen"]]
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
    # Create binary variables for each day of the week (1-7)
    for day in range(1, 8):
        features3[f'day_{day}'] = (features3['time'].dt.dayofweek + 1 == day).astype(int)
    # Create binary variables for each month of the year (1-12)
    for month in range(1, 13):
        features3[f'month_{month}'] = (features3['time'].dt.month == month).astype(int)
    # CREATE FINAL FEATURE DATASET
    features = features.merge(features2, on="time", how="left").merge(features3, on="time", how="left")
    features['date'] = pd.to_datetime(features['time']).dt.normalize()
    features = features.merge(df_oil, on="date", how="left").merge(df_msci, on="date", how="left")
    # DETERMINE CUSTOMERS
    customer_names = []
    for column_name in consumptions.columns:
        if "VALUEMWH" in column_name:
            customer_names.append("_".join(column_name.split("_")[1:]))
    # Return
    return consumptions, features, customer_names

def getDataForCustomer(customer, consumptions, features):
    Y = consumptions[["time", "VALUEMWHMETERINGDATA_"+customer]]
    Y = Y.rename(columns={"VALUEMWHMETERINGDATA_"+customer:"consumption"})
    X = features[["time", "spv", "temp", "INITIALROLLOUTVALUE_"+customer, "day_nr_inc", "is_holiday", "is_weekend", "month", "week", "day_of_week", "year", "hour", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7", "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",]]
    return Y, X

def logResiduals(residuals, output_data_path, customer, Y_train):
    Y_train["residuals"] = residuals
    Y_train = Y_train[["time", "residuals"]]
    Y_train.to_csv(output_data_path+"/"+"mean/"+customer+"_residuals.csv", index=None)

def trainModel_MEAN(Y):
    mean = np.mean(Y["consumption"])
    return mean

def doForecast_MEAN(mean):   
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast["consumption"] = mean
    return Y_forecast 

def trainModel_HOURLYMEAN(Y):
    Y['hour'] = Y['time'].dt.hour
    hourly_means = Y.groupby('hour')['consumption'].mean()
    Y_hourly_means = Y.copy()
    Y_hourly_means['true_consumption'] = Y_hourly_means['consumption'] 
    Y_hourly_means['consumption'] = Y['hour'].map(hourly_means)
    return hourly_means

def doForecast_HOURLYMEAN(hourly_means):
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast["consumption"] = Y_forecast['hour'].map(hourly_means)
    return Y_forecast 

def trainModel_HOURLYMEAN7(Y):
    Y['day_of_week'] = Y['time'].dt.dayofweek
    Y['hour'] = Y['time'].dt.hour
    weekly_hourly_means = Y.groupby(['day_of_week', 'hour'])['consumption'].mean()
    Y_hourly_means = Y.copy()
    Y_hourly_means['true_consumption'] = Y_hourly_means['consumption'] 
    Y_hourly_means['consumption'] = Y_hourly_means.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    return weekly_hourly_means

def doForecast_HOURLYMEAN7(weekly_hourly_means):
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['consumption'] = Y_forecast.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    return Y_forecast 

def trainModel_HOURLYMEAN7_GB(X_train, Y_train):
    def getResiduals_HourlyMean7(Y_train):
        weekly_hourly_means = trainModel(Y_train, X_train, modelType="hourlymean7")
        Y_fitted = Y_train.copy()
        Y_fitted['day_of_week'] = Y_fitted['time'].dt.dayofweek
        Y_fitted['hour'] = Y_fitted['time'].dt.hour
        Y_fitted['consumption'] = Y_fitted.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
        residuals_train = Y_fitted.copy()
        residuals_train["residual"] = Y_fitted["consumption"] - Y_train["consumption"]
        residuals_train = residuals_train[["time", "residual"]]
        return residuals_train, Y_fitted, weekly_hourly_means
    residuals_train, Y_fitted, weekly_hourly_means = getResiduals_HourlyMean7(Y_train)
    X_train = X_train[["spv", "temp", "day_nr_inc", "is_holiday", "is_weekend", "month", "day_of_week", "hour", "INITIALROLLOUTVALUE_"+customer]]
    grad_boost = HistGradientBoostingRegressor().fit(X_train, residuals_train["residual"])
    return [grad_boost, weekly_hourly_means]
    
def doForecast_HOURLYMEAN7_GB(model, X_test):
    grad_boost = model[0]
    weekly_hourly_means = model[1]
    X_test_sub  = X_test [["spv", "temp", "day_nr_inc", "is_holiday", "is_weekend", "month", "day_of_week", "hour", "INITIALROLLOUTVALUE_"+customer]]
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['hourlymean7'] = Y_forecast.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    Y_forecast["GB_residuals"] = grad_boost.predict(X_test_sub)
    Y_forecast["consumption"] = Y_forecast['hourlymean7'] - Y_forecast["GB_residuals"]
    return Y_forecast 

def trainModel_GRADBOOST(X,Y):
    X = X.fillna(0)
    Y = Y.fillna(0)
    grad_boost = HistGradientBoostingRegressor().fit(X.drop(['time', 'month'],axis = 1),Y['consumption'])
    return grad_boost

def doForecast_GRADBOOST(gradboost, X):
    X = X.fillna(0)
    pred_grad_boost = gradboost.predict(X.drop(['time', 'month'],axis = 1))
    forecast_time = pd.date_range(start=testing_date_range[0], end=testing_date_range[1], freq='H')
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['consumption'] = pred_grad_boost
    return Y_forecast

def trainModel(Y, X, modelType):
    if modelType=="mean":
        model = trainModel_MEAN(Y)
    elif modelType=="hourlymean":
        model = trainModel_HOURLYMEAN(Y)
    elif modelType=="hourlymean7":
        model = trainModel_HOURLYMEAN7(Y)
    elif modelType=="gradboost":
        model = trainModel_GRADBOOST(X, Y)
    elif modelType=="hourlymean7_gb":
        model = trainModel_HOURLYMEAN7_GB(X, Y)
    else:
        print("ERR UNKNOWN MODEL ", modelType)
        sys.exit(0)
    return model

def doForecast(model, modelType, X):
    if modelType=="mean":
        Y_forecast = doForecast_MEAN(model)
    elif modelType=="hourlymean":
        Y_forecast = doForecast_HOURLYMEAN(model)
    elif modelType=="hourlymean7":
        Y_forecast = doForecast_HOURLYMEAN7(model)
    elif modelType=="gradboost":
        Y_forecast = doForecast_GRADBOOST(model, X)
    elif modelType=="hourlymean7_gb":
        Y_forecast = doForecast_HOURLYMEAN7_GB(model, X)
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
model_types = ["mean", "hourlymean", "hourlymean7", "hourlymean7_gb", "gradboost"]
model_types_nonan = ["hourlymean7_gb", "gradboost"]
modelType = "hourlymean7"



consumptions, features, customer_names = loadData(input_data_path, country, training_date_from, training_date_to, date_format="%Y-%m-%d %H:%M:%S")




import sys
sys.exit(0)


"""
consumptions, features, customer_names = loadData(input_data_path, country, training_date_from, training_date_to, date_format="%Y-%m-%d %H:%M:%S")
customer = customer_names[0]


Y, X = getDataForCustomer(customer, consumptions, features)

Y_train = Y.copy()
Y_test = Y.copy()

X_train = X.copy()
X_test = X.copy()

Y_train = Y_train[(Y_train["time"] >= training_date_range[0]) & (Y_train["time"] <= training_date_range[1])]
Y_test = Y_test[(Y_test["time"] >= testing_date_range[0]) & (Y_test["time"] <= testing_date_range[1])]
X_train = X_train[(X_train["time"] >= training_date_range[0]) & (X_train["time"] <= training_date_range[1])]
X_test = X_test[(X_test["time"] >= testing_date_range[0]) & (X_test["time"] <= testing_date_range[1])]
Y_train = Y_train.reset_index()
Y_test = Y_test.reset_index()
X_train = X_train.reset_index()
X_test = X_test.reset_index()


def getResiduals_HourlyMean7(Y_train):
    weekly_hourly_means = trainModel(Y_train, modelType="hourlymean7")
    Y_fitted = Y_train.copy()
    Y_fitted['day_of_week'] = Y_fitted['time'].dt.dayofweek
    Y_fitted['hour'] = Y_fitted['time'].dt.hour
    Y_fitted['consumption'] = Y_fitted.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    residuals_train = Y_fitted.copy()
    residuals_train["residual"] = Y_fitted["consumption"] - Y_train["consumption"]
    residuals_train = residuals_train[["time", "residual"]]
    return residuals_train, Y_fitted

residuals_train, Y_fitted = getResiduals_HourlyMean7(Y_train)
residuals_test, Y_fitted2 = getResiduals_HourlyMean7(Y_test)

X_train = X_train[["spv", "temp", "day_nr_inc", "is_holiday", "is_weekend", "month", "day_of_week", "hour", "INITIALROLLOUTVALUE_"+customer]]
X_test  = X_test [["spv", "temp", "day_nr_inc", "is_holiday", "is_weekend", "month", "day_of_week", "hour", "INITIALROLLOUTVALUE_"+customer]]

# GRADIENT BOOST
grad_boost = HistGradientBoostingRegressor().fit(X_train, residuals_train["residual"])
gb_residuals_pred_train = grad_boost.predict(X_train)
gb_residuals_pred_test = grad_boost.predict(X_test)

# LINEAR REGRESSION
model = LinearRegression()
model.fit(X_train, residuals_train["residual"])
lr_residuals_pred_train = model.predict(X_train)
lr_residuals_pred_test = model.predict(X_test)

plt.figure()
plt.subplot(3,1,1)
plt.plot(Y_train["time"], Y_train["consumption"])
plt.plot(Y_fitted["time"], Y_fitted["consumption"])
plt.subplot(3,1,2)
plt.plot(residuals_train["time"], residuals_train["residual"])
plt.plot(residuals_train["time"], lr_residuals_pred_train)
plt.plot(residuals_train["time"], gb_residuals_pred_train)
plt.subplot(3,1,3)
# plt.plot(Y_train["time"], Y_train["consumption"])
plt.plot(Y_fitted["time"], np.asarray(Y_fitted["consumption"])-np.asarray(residuals_train["residual"]))
plt.plot(Y_fitted["time"], np.asarray(Y_fitted["consumption"])-np.asarray(lr_residuals_pred_train))
plt.plot(Y_fitted["time"], np.asarray(Y_fitted["consumption"])-np.asarray(gb_residuals_pred_train))


"""








# """
# INFERENCE
region_data = {}
for country in ["IT", "ES"]:
    # STEP 1: LOAD DATA
    consumptions, features, customer_names = loadData(input_data_path, country, training_date_from, training_date_to, date_format="%Y-%m-%d %H:%M:%S")
    customer = customer_names[0]
    region_true = []
    region_pred = []
    for customer in customer_names:
        # STEP 2: CLEAN DATA
        Y, X = getDataForCustomer(customer, consumptions, features)
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
        print(country, customer, modelType)
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
        model = trainModel(Y_train, X_train, modelType)
        # STEP 4: PREDICTION
        Y_forecast = doForecast(model, modelType, X_test)
        # Store Results
        region_true.append(Y_test["consumption"].tolist())
        region_pred.append(Y_forecast["consumption"].tolist())
    region_data[country] = {}
    region_data[country]["true"] = region_true
    region_data[country]["pred"] = region_pred




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