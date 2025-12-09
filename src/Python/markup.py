import os
import json
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta, MO

import polygon_data 

# @brief: Checks for holes in data
# @param data: DataFrame with datetime index
def check_missing_dates(data):
    start = data.index[0]
    end = data.index[-1]
    data_dates = pd.to_datetime(data.index)
    data_dates = data_dates.date

    nyse = mcal.get_calendar('NYSE')
    mkt_dates = nyse.valid_days(start, end)
    mkt_dates = pd.to_datetime(mkt_dates)
    mkt_dates = mkt_dates.date

    # mcal dates include timezone UTC (even though they zero out the time) which returns
    # false when compared non-timezone aware datetime. So instead we just compare the dates

    missing = np.setdiff1d(mkt_dates, data_dates)
    if len(missing) != 0:
        msg = "The are missing dates in the data! Dumping date differences:\n\n" \
             + np.array_string(missing)
        raise Exception(msg)
          
    pass


# @brief: Creates input expected by MonteCarlo.cpp. Based on given time-series and 
# frequency creates two lists: 1) the first containing the indexes of the start dates of each
# contract period, and 2) the length of each period in the time-series. This way, the cpp
# file can be naive of date and contract type
# @tseries: DataFrame of stock price series
# @frequency: The contract frequency
def build_cpp_input_file(tseries, num_points_for_fit = 24, freq = "monthly", exp_day = "friday"):
    end_date = tseries.index[-1].to_pydatetime()
    date = tseries.index[0].to_pydatetime()

    forecast_indexes = []
    forecast_horizons = []
    
    # The time-series used for daily is taken from polygon_data.pull_yahoo_data_daily().
    if freq == "daily":
        forecast_indexes = [i for i in range(0,len(tseries),2)] # Every quote at an even index is an open price
        forecast_horizons = [1]*len(forecast_indexes) # Every contract period is has length 1 in the timeseries

    if freq != "daily":
        contract_start = polygon_data.get_next_expiration(date, freq, exp_day)
        contract_end = polygon_data.get_next_expiration(contract_start, freq, exp_day)

        while contract_start <= end_date and contract_end <= end_date:
            # already checked to see if holes in data so there should be no KeyError here
            start = tseries.index.get_loc(contract_start)
            end = tseries.index.get_loc(contract_end)
            length = end - start + 1
            forecast_indexes.append(start)
            forecast_horizons.append(length)
            
            contract_start = contract_end
            contract_end = polygon_data.get_next_expiration(contract_end, freq, exp_day)

    price_series = tseries.iloc[:,0].to_list()
    date_series = [d.strftime("%Y%m%d") for d in tseries.index.to_pydatetime()]
    cpp_input = {  
        "price_series": price_series,
        "date_series": date_series, 
        "forecast_indexes": forecast_indexes,
        "forecast_horizons": forecast_horizons,
        "num_points_for_fit": num_points_for_fit
    }
    
    return cpp_input
    

def df_to_json(df, num_points_for_fit = 24, fname_out = "price_input.json",  freq = "monthly", exp_day = "friday", from_pickle = False, fname_in = ""):
    
    if from_pickle == True:
        path_in = os.path.dirname(__file__)
        fpath = os.path.join(path_in, "data", fname_in)
        df = pd.read_pickle(fpath)

    check_missing_dates(df)
    dic = build_cpp_input_file(df, num_points_for_fit, freq, exp_day)
    
    dir_path = os.path.dirname(__file__)
    path_out = os.path.dirname(dir_path)
    path_out = os.path.join(path_out, "Cpp", "data", fname_out)

    with open(path_out, "w") as f:
        json.dump(dic, f, indent = 4)

    pass

def build_input(ticker, start_date, num_points_for_fit = 3, fname_out = "price_input.json", freq = "monthly", exp_day = "friday"):
    end_date = dt.now().date()
    if freq == "monthly":
        start_date = start_date + relativedelta(months = -1 * num_points_for_fit)
        start_date = start_date + relativedelta(day = 1)
    elif freq == "weekly":
        start_date = start_date + relativedelta(weeks = -1 * num_points_for_fit)
        start_date = start_date + relativedelta(weekday = MO(-1))
    else:
        msg = "Freq must be weekly or monthly, Daily is not implemented yet"
        raise Exception(msg)
    

    df = polygon_data.pull_yahoo_data(ticker, end_date, start_date)
    df_to_json(df, num_points_for_fit, fname_out, freq, exp_day)




# This is for creating the input to test the model in Hu et al. 2020.
def build_matlab_input(fname_in, fname_out = "matlab_price_input.json", freq = "monthly", exp_day = "friday"):
    path_in = os.path.dirname(__file__)
    fpath = os.path.join(path_in, "data", fname_in)
    df = pd.read_pickle(fpath)
    check_missing_dates(df)
    dic = build_cpp_input_file(df, 0, freq, exp_day)

    t = df.index[0].to_pydatetime()
    t = t + relativedelta(days = -1)
    t2 = t + relativedelta(years = -4)
    training_data = polygon_data.pull_yahoo_data("SPX", t, t2)
    L = len(training_data)
    price_series = training_data.iloc[:,0].to_list()
    price_series.extend(dic['price_series'])
    dic['price_series'] = price_series
    dic['forecast_indexes'] = [L + i for i in dic['forecast_indexes']]

    path = os.path.dirname(path_in)
    path_out = os.path.join(path, "MATLAB", "data", fname_out)

    with open(path_out, "w") as f:
        json.dump(dic, f, indent = 4)

    pass


def main():
    filename = "monthly_test_data_1p.pkl"
    # build_matlab_input(filename)
    # pkl_to_json(filename,num_points_for_fit= 3, freq = "monthly")

    
    ticker = "SPX"
    start_date = dt.strptime("20231201","%Y%m%d") #first day of test set
    num_points_for_fit = 3
    fname_out = "price_input_weekly.json"
    freq = "weekly"
    exp_day = "friday"
    build_input(
        ticker, 
        start_date, 
        num_points_for_fit, 
        fname_out, 
        freq, 
        exp_day
    )


if __name__ == '__main__':
    main()