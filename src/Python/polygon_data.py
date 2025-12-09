import time
from datetime import datetime as dt
from datetime import timezone
import requests
from dateutil.relativedelta import relativedelta, MO,TU,WE,TH,FR
import pandas_market_calendars as mcal
import yfinance 
import pandas as pd
import httpx


POLYGON_VERSION = "paid"
API_KEY = "Place you API key here"

RELATIVE_DELTA_WEEKDAY_MAP = {"monday" : MO,
                              "tuesday" : TU,
                              "wednesday": WE,
                              "thursday": TH,
                              "friday": FR}
NUM_TO_DAY_MAP = {0:"monday",
                  1: "tuesday",
                  2: "wednesday",
                  3: "thursday",
                  4: "friday"}

class StatusCodeError(Exception):
    
    def __init__(self, msg, status_code, json_dump={}):
        self.message = msg
        self.code = status_code
        self.dump = json_dump
        super().__init__(self.message)

    def status_code(self):
        return self.code
    
    def json(self):
        return self.dump
    

class PolygonStatusError(StatusCodeError):
    
    def __init__(self, msg, status_code, json_dump={}):
        super().__init__(msg, status_code, json_dump)


class FreePolygonVersionError(Exception):
    
    def __init__(self):
        self.msg = "You are trying to do something not allowed with the FREE version of polygon!"
        super().__init__(self.msg)


# @brief: Returns the ticker of an option contract corresponding to the input parmeters.
# @param underlying: The ticker of the underlying stock or index
# @param exp_date: Expration date of the desired contract expressed as a datetime object
# @param contract_type: 'call' or 'put' for call or put
# @param price: Strike price of the option expressed as a float in dollars
def get_option_ticker(underlying, exp_date, contract_type, price, freq = "monthly"):

    assert contract_type.lower() in ["call","put"], "contract_type expected to be 'call' or 'put'"
    c_type = 'C'
    if contract_type.lower() == "put":
        c_type = 'P'


    ticker = ''


    if underlying in ["SPX", "NDX"]:
        r = 0
        base = 5
    else:
        r = 0
        base = 1

    # Poylgon contracts are rounded to nearest 5 for SPX (may need to change the base for 
    # other underlyings)
    def _round(x, base, r):
        return base * round(x/base, r)

    p = str(int(_round(price,base,r) * 1000))
    p = p.zfill(8)

    # Weekly options for SPX have 'W' added to the ticker. These are different options SPX 
    # options that expire on the same date. Generally, CME group does not list these 
    # 'weeklies' on the same friday as the monthlies expiration date (third friday). 
    # However, Polygon has OPRA quotes for option contracts (with and without 'W') that 
    # expire on the same date. Also NASDAQ options (NDX OR NDXP)do not have option 
    # contracts with this 'W' specification 
    if underlying == "SPX" and freq in ["weekly","daily"]:
        underlying = "SPXW"

    # Only set up to get quotes for PM settled NDX options which contain weekly, monthly 
    # and qurterly expirations. For NDX AM settled mobthlies (the on available frequency 
    # for AM settled NDX options), need to add functionality. Currently it's not a priorty.
    if underlying == "NDX":
        underlying = "NDXP"

    ticker = f"{underlying}{exp_date.strftime('%y%m%d')}{c_type}{p}"

    return ticker


# @brief returns the number of trding days between @start_date and @end_date
def num_trading_days(start_date, end_date):
    
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date,
                             end_date=end_date)
    return len(schedule)


# Returns the next valid market day after the given date
def get_next_trading_day(date):

    nyse = mcal.get_calendar('NYSE')
    t = date + relativedelta(days = 1)
    counter = 0

    while nyse.valid_days(t,t).empty and counter < 5:
        t = t + relativedelta(days = 1)
        counter += 1

    return t


# @brief Returns the closest trading date up to and including @param t.
#   If an expiration would be on a holiday, then the preceding trading date is the valid
#   expiration date for the option. 
# @param t: datetime.datetime
def  valid_expiration(date):

    nyse = mcal.get_calendar('NYSE')
    counter = 0
    t = date
    while nyse.valid_days(t,t).empty and counter < 5:
        t = t + relativedelta(days = -1)
        counter += 1

    if counter >= 5:
        msg  = "Something went wront with finding the excercise date of an option.\n"
        msg += f"The current date is {t.isoformat()}\n"
        msg += f"The previous working date was {date.isoformat()}\n"
        raise   Exception(msg)
    
    return t


# @brief: Returns the start of the trading period for an option contract
# @param exp_date: (datetime.datetime) the expiration date of the contract
# @param freq: (str) the frequency of the contract. Is is a "monthly","weekly","daily"
def get_start_of_contract(exp_date, freq):

    if freq == "quarterly":
        t = exp_date + relativedelta(day = 1)
        t = t + relativedelta(months=-3)
        t = t + relativedelta(weekday=FR(3))
        t = valid_expiration(t)
        return t
    
    if freq == "monthly":
        t = exp_date + relativedelta(day = 1)
        t = t + relativedelta(months = -1)
        t = t + relativedelta(weekday=FR(3))
        t = valid_expiration(t)
        return t

    if freq == "weekly":
        t = exp_date + relativedelta(weeks=-1,days=-1)
        t = valid_expiration(t)
        return t
    
    if freq == "daily":
        t = exp_date
        return t


# @brief: Monthly option contracts generally expire on the third Friday of every month.
#         Thus this returns the third Friday of the month following the passed in date.
# @param date: (datetime.datetime) object preceding the desired Friday
def get_next_expiration(date, freq, exp_day = "friday"):

    if freq == "daily":
        return get_next_trading_day(date)
    
    if freq == "weekly":
        # need to reset to exp_day in case the previous exp_date was a holiday
        t = date + relativedelta(weekday = RELATIVE_DELTA_WEEKDAY_MAP[exp_day.lower()])      
        t = t + relativedelta(days=+1, weekday = RELATIVE_DELTA_WEEKDAY_MAP[exp_day.lower()])
        t = valid_expiration(t)
        return t

    if freq == "monthly":
        t = date + relativedelta(day = 1)
        t = t + relativedelta(months = 1)
        t = t + relativedelta(weekday = FR(3))
        t = valid_expiration(t)
        return t
    
    if freq == "quarterly":
        m = date.month//3
        if date.month%3 == 0 and date >= date +relativedelta(weekday = (FR(3))):
            m = m+1
        
        t = date + relativedelta(day = 1)
        t = t + relativedelta(month = 1)
        t = t + relativedelta(months = 3*(m+1)-1)
        t = t + relativedelta(weekday=FR(3))
        valid_expiration(t)
        return t 


# @brief: Returns the first trading date and expiration date of an iption contract with 
#         the desired frequency
# @param date: (datetime.datetime) a date before the expiration of the contract
# @param freq: (str) Frequency of desired contract expiration (ex: "monthly", "weekly","daily")
# @param exp_day: (str) Weekday of the expiration day (only important for weekly options)
def get_next_contract_window(date, freq, exp_day = "Friday"):

    end_date = get_next_expiration(date, freq, exp_day)
    start_date = get_start_of_contract(end_date,freq)

    return start_date, end_date


# Returns if given date is a trading holiday
# @param date: date is expected to be datetime.datetime
def is_holiday(date):
    date = pd.to_datetime(date)
    nyse = mcal.get_calendar('NYSE')
    holidays = nyse.holidays()
    return date in holidays.holidays


# @brief: Pulls afjusted close price for previous 1000 trading days up to given date.
# @param ticker: Ticker to pull data for
# @param date: Last date in history window  
def pull_yahoo_data(ticker, end_date, start_date = None):

    # need to add upcarrot (^) to ticker if pulling an index from yahoo
    if ticker in ("SPX", "XSP", "DJI", "IXIX"):
        ticker = "^" + ticker
    
    if not start_date:
        start_date = end_date + relativedelta(years = -4) # roughly 1000 points

    end_date = end_date + relativedelta(days = 1) # yahoo data download is exclusive of the end date 
    data = yfinance.download(ticker, start_date, end_date, auto_adjust=True)
    data = data['Close'] # The close price is already auto adjusted by yfinance

    assert(len(data) > 0)

    return data


# Returns DataFrame of timeseries of prices used for 0DTE modeling. It is a single series
# of open and close prices for each day
def pull_yahoo_data_daily(ticker, end_date, start_date):

    # need to add upcarrot (^) to ticker if pulling an index from yahoo
    if ticker in ("SPX", "XSP", "DJI", "IXIX"):
        ticker = "^" + ticker

    data = yfinance.download(ticker, start_date, end_date, auto_adjust = True)
    assert(len(data) > 0)

    #Data return from yfinance.download has a zeroed-out time in the pandas datetime index.
    # Close and Open prices are in different columns. For fitting the model in CPP, we 
    # want a single timeseries of open and close prices in succession, so we just change
    # close prices datetime index to be after the open datetime index and merge. 
    # The time in the datetime index is by default (00:00:00). Rather than change 
    # datetimes to market open and close we just change close to (01:00:00) so it comes
    # after the open. The model fit is cimpletely naive of datetimes and instead just 
    # cares about timeseries order

    open = data["Open"].copy()
    close = data["Close"].copy()
    index = close.index
    func = lambda x: x.replace(hour=1)
    new_index = pd.Series(index).apply(func)
    close.index = new_index

    time_series =  pd.concat([open, close], sort=False) #Sort = False to surpress a future warning
    time_series.sort_index(inplace = True)

    return time_series

def pull_yahoo_price(ticker, date, sample_type):
    if ticker in ("SPX", "XSP", "DJI", "IXIX"):
        ticker = "^" + ticker

    end_date = date + relativedelta(days = 1) #yfinance.dowload is exclusive of end date


    # Sometimes Yahoo doesnt return any data when there is data expected. The yfinance 
    # package handles HTTP reponse codes in an opaque way so we can't do much here. Just 
    # sleep and try again until it fails 5 times.
    SUCCESS = False
    counter = 0 
    while not SUCCESS:
        
        data = yfinance.download(ticker, date, end_date, auto_adjust=True)
        data = data[sample_type]# The price is already auto adjusted by yfinance
        if len(data) > 0:
            SUCCESS = True
        else:
            print("Nothing returned from yahoo data. sleeping 20 seconds...")    
            time.sleep(20)

        counter += 1

        if counter >= 5:
            msg = f"Tried pulling open price for yahoo data on {end_date} for {ticker}, yet nothing was returned."
            raise Exception(msg)

    data = float(data.iloc[-1].iloc[-1])
    return data


# Returns the adjusted open price of a ticker on the given date
def get_yahoo_open_price(ticker, date):
    return pull_yahoo_price(ticker, date, 'Open')

def get_yahoo_close_price(ticker, date):
    return pull_yahoo_price(ticker, date, 'Close')


# @brief: Returns a truncated list of option contracts from Polygon.io All Contracts endpoint.
#         The list is centered at the current price of the underlying_ticker.
# @param underlying_ticker: Ticker of the underlying asset e.g. SPX
# @param as_of_date: (datetime.datetime) as-of-date for listed option contract quotes 
# @param exp_date: (datetime.datetime) expiration date of option contract
# @param price: (float) price of the underlying on the as_of_date
# @param contract_type: Type of option contract e.g. "call" or "put"
# @return: DataFrame of valid option contracts. Format is based on the Polygon.io REST API
#          end point https://polygon.io/docs/rest/options/contracts/all-contracts
def get_valid_strikes(underlying_ticker, as_of_date, exp_date, price, contract_type, freq = "monthly"):

    prefix = "https://api.polygon.io/v3/reference/options/contracts?"
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    exp_str = exp_date.strftime("%Y-%m-%d")
    req_str = ''.join([prefix,  
               f"underlying_ticker={underlying_ticker}&contract_type={contract_type.lower()}",
               f"&expiration_date={exp_str}&as_of={as_of_str}&order=asc&limit=1000",
               f"&sort=ticker&apiKey={API_KEY}",
                ])
    
    r = requests.get(req_str)

    if r.status_code != 200:
        msg = f"Recieved status code {r.status_code} from polygon.io API endpoint\n"
        raise StatusCodeError(msg,r.status_code,r.json())
    
    dump = r.json()
    data =pd.DataFrame(dump["results"])

    # To reduce the computational load we are truncating option contracts listed by 
    # Polygon.io which are not realistic to be bought and sold. For exampled, Polygon.io
    # lists contracts for SPX from a strike price of 200 to 11000 which are clearly not 
    # reasonable strike prices for a SPX option. For reference the S&P was appx 6,400 on 
    # 2025-08-26. Thus we truncate the list of contracts relative to the price of the 
    # underlying on the as-of-date.
    #
    # The largest monthly gain and loss  was appx 16% and 21% respectively
    # The largest daily gain and loss was appx 10% and 21% respectively
    #
    # Therefore, a window of .25 * current_price should be enough to capture all relevant options.
    # (for computational purposes I'm lowering the MARKET_THRESHOLD to .15)
    market_threshold = None
    if freq in ["monthly","weekly","daily"]:
        market_threshold = .15
    elif freq in ["quarterly"]:
        market_threshold = .25

    assert market_threshold is not None, "OOPS sometheting went wrong here with the frequency"
    
    max_price = price + market_threshold * price
    min_price = price - market_threshold * price

    
    realistic_data = data[(data['strike_price'] <= max_price) &\
                              (data['strike_price'] >= min_price)]
    
    
    # Need to reduce data load for SPX options. Options can be specified up to 5 points on
    # the index which is not necessary for our purpose. 
    if underlying_ticker == 'SPX':
        cond = realistic_data["ticker"].str.contains('W')
        if freq == "monthly":
            cond = ~cond

        realistic_data = realistic_data.loc[(realistic_data["strike_price"]%10 == 0) & (cond)]
       
    return realistic_data.set_index('strike_price')

def custom_bar_query(ticker, from_dt, to_dt, freq = "monthly"):
    from_dt = from_dt.strftime("%Y-%m-%d")
    to_dt = to_dt.strftime("%Y-%m-%d")
    prefix = "https://api.polygon.io/v2/aggs"
    query_params = "adjusted=true&sort=asc"
    
    # if freq in ["monthly","weekly"]:
    range = 1
    ts_frequency = "day" 
    if freq == "daily":
        range = 10
        ts_frequency = "minute"
    

    req_str = f"{prefix}/ticker/O:{ticker}/range/{range}/{ts_frequency}/{from_dt}/{to_dt}?{query_params}&apiKey={API_KEY}"
    return req_str

# @brief: Pulls option contract data from Polygon.io REST API
# @param ticker: Ticker of option contract being requested
# @param from_dt: Starting date of requested time series
# @param to_dt: Last date in requested time series (inclusive). Usually assumed to be the 
#               expiration date of the option
#
# For more information see Polygon REST API docs at 
# https://polygon.io/docs/rest/options/aggregates/custom-bars
async def pull_option(ticker, from_dt , to_dt, freq = "monthly", client:httpx.AsyncClient= None):

    req_str = custom_bar_query(ticker, from_dt, to_dt, freq)

    if client :
        r = await client.get(req_str)

    else:
        r = httpx.get(req_str)

    if r.status_code != 200:
        msg = f"Recieved status code {r.status_code} from polygon.io API endpoint\n"
        raise StatusCodeError(msg, r.status_code,r.json())

    dump = r.json()
    if dump['status'] != 'OK':
        msg = f"Recieved non-OK status code from Polygon response.\n Status recieved: {dump['status']}\n"
        msg = f"Dumping non-timeseries response for investigation:\n"
        dump.pop('results',None)
        for k in dump.keys():
            msg+= f"{k} : {dump[k]}\n"
        raise PolygonStatusError(msg,dump['status'],dump)

    return dump    

#TODO UPDATE brief and return
# @brief: Converting millisecond epochs to datetimes and converts data fromm HTTP-request 
#         json dumps to dataframe.
# @param data: "results" field return by polygon option endpoint. It's arranged as a list
#              of dictionaries with each list element being a different time point. 
# @param start_date: Start of window of pulled dates, expted type is datetime.datetime
# @param end_date: End of window of pulled dates, expted type is datetime.datetime
def clean_data(data, start_date, end_date):

    close_prices = []
    dates = []
    volume = [] #really just keeping volumne for debug purposes 

    # debug
    tic = dt.now()

    for row in data:
        close_prices.append(row['c'])
        volume.append(row['v'])

        # The time entry in the Polygon response is epoch time in milliseconds while
        # datetime.fromtimestamp expects epoch time in seconds
        ms = row['t']
        t = dt.fromtimestamp(ms/1e3, timezone.utc) # timezone needs to be UTC or you get the wrong date
        dates.append(pd.to_datetime(t)) # pandas datetime indexing doesnt work well with datetime.datetime objects. Thus converting to str

    cleaned_data = pd.DataFrame(data = {'timestamp': dates, 'close_price': close_prices, 'volume':volume})
    cleaned_data = cleaned_data.set_index('timestamp')

    return cleaned_data

def main():
    print("Pulling from Polygon REST API")

    TERMINATION_DATE = dt.today()
    start_date = get_next_expiration(dt.strptime("20250102","%Y%m%d"))
    end_date = get_next_expiration(start_date)

    underlying_ticker = "SPX"
    option_type = 'call' 

    print(f"Pulling data for {underlying_ticker} \n \
          between {start_date} and {end_date}.\n")

    monthlies_ts = []
    expirations = []

    while end_date <= TERMINATION_DATE:        
        
        current_price = get_yahoo_open_price(underlying_ticker, start_date)
        option_ticker = get_option_ticker(underlying_ticker, end_date, option_type, current_price)
        print(f"Pulling data for {option_ticker}")
        data = pull_option(option_ticker,start_date,end_date)
        data = data['results']
        data = clean_data(data, start_date, end_date)
        monthlies_ts.append(data)
        expirations.append(pd.to_datetime(end_date))

        print(f"Data pull for {option_ticker} complete!")

        start_date = get_next_trading_day(end_date)
        end_date = get_next_expiration(end_date)

        # The free Polygon.io version only allows 5 API calls per min 
        if POLYGON_VERSION.upper() == 'FREE':
            time.sleep(20)
    
    monthlies_df = pd.concat(monthlies_ts, keys = expirations, names = ['expiration']) 
    
    print(f"JOB FINISHED! \n Completed pulling data for {underlying_ticker} \n \
          between {start_date} and {end_date}.\n")
    
    print("Dumping dataframe to pickle.")
    monthlies_df.to_pickle(f"{underlying_ticker}_monthlies.pkl")
    
    print("Pickle saved!")
    pass


if __name__ == "__main__":
    main()