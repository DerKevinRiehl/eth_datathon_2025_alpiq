import yfinance as yf
import pandas as pd

# Define the ticker symbol for WTI crude oil futures
ticker = "CL=F"  # WTI Crude Oil Futures

# Define the date range
start_date = "2022-01-01"
end_date = "2024-09-01"

# Fetch the historical data
wti_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Extract only the 'Open' column
wti_open_prices = wti_data[['Open']]

# Save the open prices to a CSV file
output_file = "wti_open_prices.csv"
wti_open_prices.to_csv(output_file)

print(f"WTI oil futures open prices saved to {output_file}")