# IMPORTS
import pandas as pd
from data import DataLoader
from os.path import join
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX




# PARAMETERS
path = "../../data/1_original/OneDrive_2025-04-05/Alpiq ETHdatathon challenge 2025/datasets2025/"
countries = ["IT", "ES"]
country = "IT"
training_date_from = pd.Timestamp("2022-01-01 00:00:00")
training_date_to = pd.Timestamp("2022-05-31 23:00:00")
# training_date_to = pd.Timestamp("2024-07-31 23:00:00")

# LOAD DATA
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

consumptions, features, customer_names = loadData(path, country, training_date_from, training_date_to, date_format="%Y-%m-%d %H:%M:%S")

# Get Time Series Date For Specific Customer
customer = customer_names[0]
Y = consumptions[["time", "total_consumption"]]
X = features[["time", "spv", "temp", "INITIALROLLOUTVALUE_"+customer, "day_nr_inc", "is_holiday", "is_weekend", "month", "week", "day_of_week", "year", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7", "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",]]



# Determine Daily SARIMAX (24h)
# Define SARIMAX parameters
p, d, q = 2, 1, 2   # Non-seasonal parameters (example)
P, D, Q, s = 1, 1, 1, 24  # Seasonal parameters (24-hour seasonality)
# Fit SARIMAX model
model = SARIMAX(Y['total_consumption'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)
# Print model summary
print(model_fit.summary())

# Forecast next 24 hours (example)
forecast_steps = 24
forecast = model_fit.forecast(steps=forecast_steps)
forecast_time = pd.date_range(start=training_date_to, periods=forecast_steps + 1, freq='H')[1:]
Y_forecast = pd.DataFrame({'time': forecast_time})
Y_forecast["Y"] = forecast.tolist()

# Plot forecast vs original data
plt.figure(figsize=(12, 6))
plt.plot(Y["time"], Y['total_consumption'], label='Historical Data')
plt.plot(Y_forecast["time"], Y_forecast["Y"], label='Forecast', color='red')
plt.title('SARIMAX Forecast')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()


