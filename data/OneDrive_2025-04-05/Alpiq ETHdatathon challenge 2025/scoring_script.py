import pandas as pd
import numpy as np
from os.path import join
from datetime import datetime

date_format = '%Y-%m-%d %H:%M:%S'
student_path = "outputs"
testing_set_path = r"datasets2025"

absolute_error = {}
portfolio_error = {} 

for team_name in ['X']:

    try:
        for country in ['ES', 'IT']:
            student_solution_path = student_path+"/students_results_" + team_name + '_' + country + ".csv"
            student_solution = pd.read_csv(
                student_solution_path, index_col=0, parse_dates=True, date_format=date_format)
            testing_set_fullpath = join(
                testing_set_path, "example_set_" + country + ".csv")
            testing_set = pd.read_csv(
                testing_set_fullpath, index_col=0, parse_dates=True, date_format=date_format)

            country_error = (student_solution - testing_set).abs().sum().sum()
            portfolio_country_error = (student_solution - testing_set).sum(axis=1).abs().sum() 

            assert np.all(student_solution.columns ==
                          testing_set.columns), 'Wrong header or header order for team ' + team_name
            assert np.all(student_solution.index ==
                          testing_set.index), 'Wrong index or index order for team ' + team_name
            assert isinstance(
                country_error, np.float64), 'Wrong error type for team ' + team_name
            assert student_solution.isna().sum().sum(
            ) == 0, 'NaN in forecast for team ' + team_name

            absolute_error[country] = country_error
            portfolio_error[country] = portfolio_country_error

        forecast_score = (
            1.0*absolute_error['IT'] + 5.0*absolute_error['ES'] + 
            10.0*portfolio_error['IT'] + 50.0*portfolio_error['ES']
        )
        print('The team ' + team_name + ' reached a forecast score of ' +
              str(np.round(forecast_score, 0)))
    except Exception as e:
        print('Error for team ' + team_name)
        print(e)
print('=== End of the script, %s. ===' % (str(datetime.now())))
