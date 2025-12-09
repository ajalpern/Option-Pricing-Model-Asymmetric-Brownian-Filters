import time
from datetime import datetime as dt
import os
import asyncio
import logging

from dateutil.relativedelta import relativedelta
import pandas as pd
import httpx

import polygon_data

logger = logging.getLogger()
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.basicConfig(format="%(asctime)s %(message)s", 
                    level = logging.INFO)

async def _range_generator(N):
    for i in range(N):
        yield i


async def _pull_option_async(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, task_id, ticker, from_dt , to_dt, strike, freq, retry_count = 0):

    # Need try block around (as opposed to vice versa) to catch asyncio.CancelledException
    # in the case that the task is cancelled before it even starts. Which is common in this
    # implementation to get around Polygon.io's poor REST API 
    try:
        async with semaphore:

            result = await asyncio.to_thread(polygon_data.pull_option, 
                                             ticker, 
                                             from_dt, 
                                             to_dt,
                                             freq , 
                                             client,)
            data = await result 

            # Set threshold for the amount of trades for a given contract. If there are 
            # less records than the threshold, the option contract data is ignored
            threshold = 4
            if freq == "weekly":
                threshold = 2

            if data['resultsCount'] <= threshold:
                return None, None
                # await _try_to_cancel(task_id)

            return data, strike
                    
    except asyncio.CancelledError as e:
        data = {}
        return data, strike
    
    except polygon_data.PolygonStatusError as e:
        logger.error(f"Non-OK polygon status error! status = {e.status_code()}")
        data = {}
        return data, strike
    
    except httpx.TimeoutException as e:
        
        # TODO: maybe make retry_count variable in the future, but for now this works 
        # totally fine
        if retry_count < 1:
            retry_count +=1

            await _pull_option_async(client, semaphore, task_id
                                    ,ticker, from_dt, strike, freq
                                    ,retry_count)
        else:
            logger.warning("Caught httpx.ReadTimeout exception. "
                           +"Retried with no success... moving on. "
                           +"If this keeps on happening, "
                           +"investigate the connection or the query")
        data = {}
        return data, strike
             
    # Cant have a finally block return data because it will cause an UnboundLocalError.
    # Furthermore, initializing result=None before the try block will always return None.
    # Thus each try/except block needs to initialize data and return it.

    

# TODO add asychronous requests to speed this up for the paid Polygon.io version  
#
# @brief: Pulling historical options data from Polygon.io is only per ticker. The ticker
# is specific to the strike price so we need to determine relevant strikes to pull and 
# then pull them all separately. Relevance for a option contract is determined by whether 
# the price movement is probable (i.e. has ever happened in the history of the market), 
# and are there any quotes for that stike.
# @param underlying_ticker: (str) Ticker of the underlying e.g. SPX, AMZN
# @param from_dt: (datetime.datetime) Beginning of option contract window
# @param to_dt: (datetime.datetime) expiration of the contract
# @param contract_type: type of option contract. e.g. "call" or "put"
# @param freq: type of option contract frequenecy of expiration e.g. ["quarterly","monthly","weekly","daily"]
# @return: Multilevel indexed dataframe for contracs with the same expiration.
#          index levels [timestamp, strike]
async def pull_option_chain(underlying_ticker, from_dt, to_dt, contract_type, current_price, freq = "monthly"):

    async def pull_options(valid_contracts, underlying_ticker, from_dt, to_dt, contract_type, freq):

        if polygon_data.POLYGON_VERSION.lower() != 'paid':
            raise polygon_data.FreePolygonVersionError()
        
        contract_data = []
        _OPTION_TASK_LIST = {}
        semaphore = asyncio.Semaphore(10)
        async with httpx.AsyncClient() as client:
            task_id = 0
            async for i in _range_generator(len(valid_contracts.index)):
                strike = valid_contracts.index[i]
                ticker = polygon_data.get_option_ticker(underlying_ticker
                                                        ,to_dt
                                                        ,contract_type
                                                        ,strike
                                                        ,freq)
                _OPTION_TASK_LIST[task_id] = asyncio.create_task(
                                                            _pull_option_async(
                                                                client
                                                                ,semaphore 
                                                                ,task_id 
                                                                ,ticker 
                                                                ,from_dt 
                                                                ,to_dt
                                                                ,strike
                                                                ,freq))
                task_id += 1
            
            
            results = await asyncio.gather(*_OPTION_TASK_LIST.values())
            
        logger.info('Cleaning data')
        contract_data = []
        strikes = []
        for r, s in results:
            if r:
                if r['resultsCount'] > 2:
                    data = r["results"]
                    data = polygon_data.clean_data(data, from_dt, to_dt)
                    contract_data.append(data)
                    strikes.append(s)

        _OPTION_TASK_LIST = {}

        return contract_data, strikes

    valid_contracts = polygon_data.get_valid_strikes(underlying_ticker, 
                                        from_dt, 
                                        to_dt, 
                                        current_price, 
                                        contract_type, 
                                        freq)
    
    out_of_money = valid_contracts.loc[valid_contracts.index > round(current_price)].copy(deep = True)
    out_of_money = out_of_money.sort_index(ascending = True)
    out_of_money_data, oom_strikes = await pull_options(out_of_money, 
                                underlying_ticker,
                                from_dt, 
                                to_dt, 
                                contract_type,
                                freq)

    in_the_money = valid_contracts.loc[valid_contracts.index <= round(current_price)].copy(deep = True)
    in_the_money = in_the_money.sort_index(ascending = False)
    in_the_money_data, itm_strikes = await pull_options(in_the_money, 
                                underlying_ticker,
                                from_dt, 
                                to_dt, 
                                contract_type,
                                freq)
    
    option_chain = pd.concat(out_of_money_data + in_the_money_data, keys = oom_strikes+itm_strikes, names= ["strike"])
    
    return option_chain.sort_index()

def main():
    freq = "monthly"
    LOOP_START_DATE = dt.strptime("20240101","%Y%m%d")
    
    TERMINATION_DATE = dt.now() + relativedelta(hour=0, minute=0, second=0, microsecond=0) 
    underlying_ticker = "SPX"
    option_type = 'call' 
    
    start_date, end_date = polygon_data.get_next_contract_window(LOOP_START_DATE, freq)
    
    logger.info("Starting Option Pull.")
    logger.info(f"Pulling {freq} data for {underlying_ticker} option contracts between\n"\
                +f"between {LOOP_START_DATE} and {TERMINATION_DATE}.")
    
    time_series = []
    expirations = []
    while end_date < TERMINATION_DATE:
        current_price = polygon_data.get_yahoo_open_price(underlying_ticker, start_date)
        data =  asyncio.run(     
                            pull_option_chain(underlying_ticker,
                                              start_date,
                                              end_date,
                                              option_type, 
                                              current_price,
                                              freq)
                            )
        time_series.append(data)
        expirations.append(pd.to_datetime(end_date))
        logger.info(f"Completed pulling data for {underlying_ticker} options "\
                   +f"with freq={freq} and expiration={end_date.date()}!")

        start_date, end_date = polygon_data.get_next_contract_window(end_date, freq)

    logger.info("Completed pulling all data!")    
    time_series_df = pd.concat(time_series, keys = expirations, names = ['expiration'])
    logger.info("Dumping dataframe to pickle.")
    data_dir = "data"
    fname = f"{underlying_ticker}_{freq}.pkl"
    fout_path = os.path.join(data_dir,fname)
    time_series_df.to_pickle(fout_path)

if __name__ == '__main__':
    main()