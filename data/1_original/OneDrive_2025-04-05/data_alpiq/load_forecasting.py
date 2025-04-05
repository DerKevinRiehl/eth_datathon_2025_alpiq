import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

# depending on your IDE, you might need to add datathon_eth. in front of data
from data import DataLoader, SimpleEncoding

# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import SimpleModel


def main(zone: str):
    """

    Train and evaluate the models for IT and ES

    """

    # Inputs
    input_path = r"datasets2025"
    output_path = r"outputs"

    # Load Datasets
    loader = DataLoader(input_path)
    # features are holidays and temperature
    training_set, features, example_results = loader.load_data(zone)
    print(training_set.head(10))


    """
    EVERYTHING STARTING FROM HERE CAN BE MODIFIED.
    """
    team_name = "OurCoolTeamName"
    # Data Manipulation and Training
    start_training = training_set.index.min()
    end_training = training_set.index.max()
    start_forecast, end_forecast = example_results.index[0], example_results.index[-1]

    range_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq="1H")
    forecast = pd.DataFrame(columns=training_set.columns, index=range_forecast)
    import matplotlib.dates as mdates

    # Find first and last entry per customer
    customer_data_range = {}

    for customer in training_set.columns:
        non_null_times = training_set.index[training_set[customer].notna()]
        if not non_null_times.empty:
            customer_data_range[customer] = {
                "first_entry": non_null_times.min(),
                "last_entry": non_null_times.max(),
            }
        else:
            customer_data_range[customer] = {
                "first_entry": None,
                "last_entry": None,
            }

    # Convert to DataFrame and save
    customer_data_range_df = pd.DataFrame.from_dict(customer_data_range, orient="index")
    customer_data_range_df.index.name = "Customer"
    customer_data_range_df.to_csv(join(output_path, f"customer_entry_range_{zone}.csv"))

    # create a dataframe with a row for each day and a column that counts how many customers give data in that day
    counter_df = pd.DataFrame(
        index=pd.date_range(start=training_set.index.min(), end=training_set.index.max(), freq="1D"),
        columns=["count"],
    )


    # default value for count is 0
    counter_df["count"] = 0
    # for each customer, check if there is data for that day and increment the counter


    counter_here = 0
    for customer in training_set.columns:
        counter_here += 1
        if counter_here < 1000000:  #mechanism to turn off at some point
            print("in iteration: ", counter_here)
            non_null_times = training_set.index[training_set[customer].notna()]
            if not non_null_times.empty:
                for time in non_null_times:
                    daily_time = time.floor('D')  # Align to daily frequency
                    counter_df.loc[daily_time, "count"] += 1



    # divide all the count values 24 to get the average number of customers that give data in that day
    counter_df["count"] = counter_df["count"] / 24
    print(counter_df.head(10))

    # save the dataframe to csv
    counter_df.to_csv(join(output_path, f"customer_data_availability_{zone}.csv"))

    # plot the number of customers that give data in that day
    fig, ax = plt.subplots()
    ax.plot(counter_df.index, counter_df["count"], 'o', ms=2)
    # format the x-axis with date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    # add grid
    ax.grid()
    # add legend
    ax.legend(["Number of customers that give data"])
    # add title
    ax.set_title("Number of customers that give data")
    # add x and y axis labels
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of customers")
    plt.savefig(join(output_path, "customer_data_availability.png"))
    plt.close(fig)


    

    # print for quick feedback
    print("\nCustomer data availability (first & last timestamps):")
    print(customer_data_range_df.head(10))

    #save customer data range to csv
    customer_data_range_df.to_csv(join(output_path, "customer_data_range.csv"))    



    counter = 0
    for costumer in training_set.columns.values:
        counter += 1
        consumption = training_set.loc[:, costumer]
        # create plots and save them to the output folder
        if counter < 20:
            print(counter, costumer)
            fig, ax = plt.subplots()
            ax.plot(consumption.index, consumption.values, 'o', ms=2)
            # format the x-axis with date labels
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            # add grid
            ax.grid()
            # add legend
            ax.legend([costumer])
            # add title
            ax.set_title(costumer)
            # add x and y axis labels
            ax.set_xlabel("Time")
            ax.set_ylabel("Consumption")
            plt.savefig(join(output_path, costumer + ".png"))
            plt.close(fig)


        
        feature_dummy = features['temp'].loc[start_training:]

        encoding = SimpleEncoding(
            consumption, feature_dummy, end_training, start_forecast, end_forecast
        )

        feature_past, feature_future, consumption_clean = (
            encoding.meta_encoding()
        )

        # Train
        model = SimpleModel()
        model.train(feature_past, consumption_clean)

        # Predict
        output = model.predict(feature_future)
        forecast[costumer] = output

    """
    END OF THE MODIFIABLE PART.
    """
    #create dataframe to hold data about the companies

    # test to make sure that the output has the expected shape.
    dummy_error = np.abs(forecast - example_results).sum().sum()
    assert np.all(forecast.columns == example_results.columns), (
        "Wrong header or header order."
    )
    assert np.all(forecast.index == example_results.index), (
        "Wrong index or index order."
    )
    assert isinstance(dummy_error, np.float64), "Wrong dummy_error type."
    assert forecast.isna().sum().sum() == 0, "NaN in forecast."
    # Your solution will be evaluated using
    # forecast_error = np.abs(forecast - testing_set).sum().sum(),
    # and then doing a weighted sum the two portfolios:
    # score = forecast_error_IT + 5 * forecast_error_ES

    forecast.to_csv(
        join(output_path, "students_results_" + team_name + "_" + country + ".csv")
    )


if __name__ == "__main__":
    country = "ES"  # it can be ES or IT
    main(country)
