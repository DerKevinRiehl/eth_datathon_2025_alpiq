import pandas as pd
from sklearn.preprocessing import Normalizer

# Load the dataset
file_path = "../../data/1_original/OneDrive_2025-04-05/Alpiq ETHdatathon challenge 2025/datasets2025/historical_metering_data_ES.csv"  # Update this path if needed
data = pd.read_csv(file_path)

# Fill empty cells (NaN) with 0
data = data.fillna(0)

# Drop the DATETIME column if it exists
if 'DATETIME' in data.columns:
    data = data.drop(columns=['DATETIME'])

# Convert all columns to numeric, coercing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Normalize the rows using sklearn's Normalizer
normalizer = Normalizer(norm='l1')  # 'l1' ensures row sums equal 1
normalized_data = normalizer.fit_transform(data)

# Convert the normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

# Check that each row sums to 1
for index, row in normalized_df.iterrows():
    row_sum = row.sum()
    assert abs(row_sum - 1) < 1e-6, f"Row {index} does not sum to 1 (sum = {row_sum})"

# Save the results to a CSV file
output_file = "contributions_normalized.csv"
normalized_df.to_csv(output_file, index=False)

print(f"Row-wise normalized contributions have been saved to {output_file}")