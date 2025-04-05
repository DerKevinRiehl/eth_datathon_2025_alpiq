# IMPORTS
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from multiprocessing import Pool




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
    # Create binary variables for each day of the week (1-7)
    for day in range(1, 8):
        features3[f'day_{day}'] = (features3['time'].dt.dayofweek + 1 == day).astype(int)
    # Create binary variables for each month of the year (1-12)
    for month in range(1, 13):
        features3[f'month_{month}'] = (features3['time'].dt.month == month).astype(int)
    # CREATE FINAL FEATURE DATASET
    features = features.merge(features2, on="time", how="left").merge(features3, on="time", how="left")
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
    X = features[["time", "spv", "temp", "INITIALROLLOUTVALUE_"+customer, "day_nr_inc", "is_holiday", "is_weekend", "month", "week", "day_of_week", "year", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7", "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",]]
    return Y, X

def analyseDataConsistency(Y):
        # Find the first non-NaN value in 'consumption'
    first_non_nan_index = Y['consumption'].first_valid_index()
    first_non_nan_date = Y.loc[first_non_nan_index, 'time']
        # Check if there are any NaN values later in the column
    has_nan_later = Y['consumption'].isnull().any()
    return first_non_nan_date, has_nan_later

def trainModel_HOURLYMEAN7(Y):
    Y['day_of_week'] = Y['time'].dt.dayofweek
    Y['hour'] = Y['time'].dt.hour
    weekly_hourly_means = Y.groupby(['day_of_week', 'hour'])['consumption'].mean()
    Y_hourly_means = Y.copy()
    Y_hourly_means['true_consumption'] = Y_hourly_means['consumption'] 
    Y_hourly_means['consumption'] = Y_hourly_means.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)

    residuals = Y_hourly_means['consumption']-Y_hourly_means['true_consumption'] 
    return weekly_hourly_means, residuals

def doForecast_HOURLYMEAN7(weekly_hourly_means, forecast_steps):
    forecast_time = pd.date_range(start=training_date_to, periods=forecast_steps + 1, freq='H')[1:]
    Y_forecast = pd.DataFrame({'time': forecast_time})
    Y_forecast['day_of_week'] = Y_forecast['time'].dt.dayofweek
    Y_forecast['hour'] = Y_forecast['time'].dt.hour
    Y_forecast['Y'] = Y_forecast.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)
    return Y_forecast 

def evaluationScore_HOURLYMEAN7(residuals):
    return np.abs(residuals).sum()

def logEvaluation(customer, evaluation_score):
    f = open(output_data_path+"hourlymean7/"+customer+"_evaluation.csv", "w+")
    f.write(str(evaluation_score))
    f.close()

def logResiduals(residuals, output_data_path, customer, Y_train):
    Y_train["residuals"] = residuals
    Y_train = Y_train[["time", "residuals"]]
    Y_train.to_csv(output_data_path+"/"+"hourlymean7/"+customer+"_residuals.csv", index=None)

# PARAMETERS
input_data_path = "../../../data/1_original/OneDrive_2025-04-05/Alpiq ETHdatathon challenge 2025/datasets2025/"
output_data_path = "../../../data/2_processed/"
countries = ["IT", "ES"]
training_date_from = pd.Timestamp("2022-01-01 00:00:00")
# training_date_to = pd.Timestamp("2022-05-31 23:00:00")
training_date_to = pd.Timestamp("2024-07-31 23:00:00")
forecast_steps = 24*31
max_threads = 4  # Maximum number of threads to run in parallel
country = "IT"
country = "ES"




# STEP 1: LOAD DATA
consumptions, features, customer_names = loadData(input_data_path, country, training_date_from, training_date_to, date_format="%Y-%m-%d %H:%M:%S")
customer = customer_names[0]






"""
# STEP 2: CLEAN DATA
Y, X = getDataForCustomer(customer, consumptions, features)


# STEP 3: TRAIN MODEL
Y_train = Y.copy()


Y['day_of_week'] = Y['time'].dt.dayofweek
Y['hour'] = Y['time'].dt.hour
weekly_hourly_means = Y.groupby(['day_of_week', 'hour'])['consumption'].mean()
Y_hourly_means = Y.copy()
Y_hourly_means['true_consumption'] = Y_hourly_means['consumption'] 
Y_hourly_means['consumption'] = Y_hourly_means.apply(lambda row: weekly_hourly_means[row['day_of_week'], row['hour']], axis=1)

residuals = Y_hourly_means['consumption']-Y_hourly_means['true_consumption'] 


Y_forecast = doForecast_HOURLYMEAN7(weekly_hourly_means, forecast_steps)

# Plot forecast vs original data
plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.title("DATA AND PREDICTION")
plt.plot(Y["time"], Y['consumption'], label='Historical Data')
plt.plot(Y_hourly_means["time"], Y_hourly_means['consumption'], label='model')
plt.plot(Y_forecast["time"], Y_forecast["Y"], label="Prediction")
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.subplot(2,1,2)
plt.plot(Y["time"], residuals)
# plt.plot(Y_forecast["time"], Y_forecast["Y_model2"], label='Forecast', color='red')
plt.title("RESIDUAL ERROR OF SARIMAX MODEL (DAILY)")
plt.xlabel('Time')
# plt.subplot(3,1,3)
# plt.plot(Y["time"], residuals2)
# plt.title("RESIDUAL ERROR OF SARIMAX MODEL (WEEKLY)")
# plt.xlabel('Time')
plt.tight_layout()
print(np.sum(np.abs(residuals)))
"""


def process_customer(customer, consumptions, features, forecast_steps, output_data_path):
    try:
        # STEP 2: CLEAN DATA
        Y, X = getDataForCustomer(customer, consumptions, features)

        # STEP 3: TRAIN MODEL
        Y_train = Y.copy()
        model, residuals = trainModel_HOURLYMEAN7(Y_train)
        evaluation_score = evaluationScore_HOURLYMEAN7(residuals)
        logEvaluation(customer, evaluation_score)
        logResiduals(residuals, output_data_path, customer, Y_train)
            
        # STEP 4: PREDICTION
        Y_forecast = doForecast_HOURLYMEAN7(model, forecast_steps)
        Y_forecast.to_csv(output_data_path+"hourlymean7/"+customer+"_prediction.csv", index=None)

        print(f"Processed customer: {customer}")
    except Exception as e:
        print(f"Error processing customer {customer}: {e}")
                
# SEQUENTIAL:
for customer in customer_names:
    process_customer(customer, consumptions, features, forecast_steps, output_data_path)

