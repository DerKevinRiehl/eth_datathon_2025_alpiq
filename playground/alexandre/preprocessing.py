import pandas as pd
import numpy as np

def process_time_series(file_path, column_index):
    """
    Process a time series dataset by handling NaNs and outliers with interpolation.
    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        column_index (int): Index of the column to process (excluding the DATETIME column).

    Returns:
        pd.DataFrame: DataFrame containing the original and cleaned time series.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Extract the desired column(s)
    selected_columns = data.iloc[:, [0, column_index]]  # Extract DATETIME and the specified column

    # Find the first non-NaN index in the selected column
    first_non_nan_index = selected_columns.iloc[:, 1].first_valid_index()

    # Trim the DataFrame to start from the first non-NaN value
    trimmed_columns = selected_columns.loc[first_non_nan_index:].reset_index(drop=True)

    # Convert the DATETIME column to pandas datetime format
    trimmed_columns.iloc[:, 0] = pd.to_datetime(trimmed_columns.iloc[:, 0], errors='coerce')

    # Perform interpolation to fill NaN values
    trimmed_columns.iloc[:, 1] = trimmed_columns.iloc[:, 1].interpolate(method='linear')

    # Define a function for moving window outlier detection
    def detect_outliers_moving_window(data, window_size=10):
        # Compute moving mean and moving standard deviation
        moving_mean = data.rolling(window=window_size).mean()
        moving_std = data.rolling(window=window_size).std()

        # Identify outliers as values more than 1.7 standard deviations from the moving mean
        outliers_mask = abs(data - moving_mean) > 1.7 * moving_std
        return outliers_mask

    # Apply moving window outlier detection
    outlier_mask = detect_outliers_moving_window(trimmed_columns.iloc[:, 1], window_size=10)

    # Count the number of outliers
    num_outliers = outlier_mask.sum()
    print(f"Number of outliers detected: {num_outliers}")

    # Replace detected outliers with NaN
    trimmed_columns['cleaned_values'] = trimmed_columns.iloc[:, 1].where(~outlier_mask, np.nan)

    # Interpolate to replace outliers with interpolated values
    trimmed_columns['cleaned_values'] = trimmed_columns['cleaned_values'].interpolate(method='linear')

    # Return only the cleaned column
    return trimmed_columns[['DATETIME', 'cleaned_values']]