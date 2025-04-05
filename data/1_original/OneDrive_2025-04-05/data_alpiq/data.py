import pandas as pd
from os.path import join


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        current_folder_location = __file__
        print(current_folder_location)
        folder_path = current_folder_location[: current_folder_location.rfind("/")] + "/datasets2025"

        consumptions_path = join(folder_path, "historical_metering_data_" + country + ".csv")
        features_path = join(folder_path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(folder_path, "example_set_" + country + ".csv")

        print("Loading data...")
        print(f"Loading consumptions from {consumptions_path}")
        print(f"Loading features from {features_path}")
        print(f"Loading example solution from {example_solution_path}")

        print("Loading consumptions...")
        consumptions = pd.read_csv(
            consumptions_path, index_col=0, parse_dates=True, date_format=date_format
        )
        print("Loading example solution...")
        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        print("Loading features...")
        features = pd.read_excel(
            features_path,
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )


        return consumptions, features, example_solution


# Encoding Part


class SimpleEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        end_training,
        start_forecast,
        end_forecast,
    ):
        self.consumption_mask = ~consumption.isna()
        self.consumption = consumption[self.consumption_mask]
        self.features = features
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast

    def meta_encoding(self):
        """
        This function returns the feature, split between past (for training) and future (for forecasting)),
        as well as the consumption, without missing values.
        :return: three numpy arrays

        """
        features_past = self.features[: self.end_training].values.reshape(-1, 1)
        features_future = self.features[
            self.start_forecast : self.end_forecast
        ].values.reshape(-1, 1)

        features_past = features_past[self.consumption_mask]

        return features_past, features_future, self.consumption
