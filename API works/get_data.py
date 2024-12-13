#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import requests
import typing
import json
import datetime

class AVParameters(typing.TypedDict, total=False):
    """
    Valid query parameters for the AlphaVantage API.
    """

    adjusted: typing.Literal["true", "false"]
    apikey: typing.Required[str]
    outputsize: typing.Literal["compact", "full"]
    datatype: typing.Literal["csv", "json"]
    extended_hours: typing.Literal["true", "false"]
    function: typing.Literal[
        "BALANCE_SHEET",
        "FEDERAL_FUNDS_RATE",
        "TIME_SERIES_INTRADAY",
        "TIME_SERIES_DAILY",
        "TIME_SERIES_DAILY_ADJUSTED",
        "TIME_SERIES_WEEKLY",
        "TIME_SERIES_WEEKLY_ADJUSTED",
        "TIME_SERIES_MONTHLY",
        "TIME_SERIES_MONTHLY_ADJUSTED",
        "GLOBAL_QUOTE",
        "SYMBOL_SEARCH",
        "MARKET_STATUS",
        "CPI",
        "UNEMPLOYMENT",
        "TREASURY_YIELD"
    ]
    interval: typing.Literal[
        "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"
    ]
    maturity: typing.Literal[
        "1month", "3month", "2year", "5year", "7year", "10year"
    ]
    month: str
    symbol: str
    maturity: str
    


def get_alphavantage(params: AVParameters, use_cache: bool = True) -> typing.Any:
    """
    Makes an HTTP GET request to the AlphaVantage API, with caching.

    Parameters:
        params (AVParameters): The query parameters to include in the request
        use_cache (bool): Whether to use a cached response

    Returns:
        response (typing.Any): A JSON response
    """
    CACHE_DIR = "/tmp/fina4460"
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    params_str = "_".join(
        f"{key}={value}" for key, value in sorted(params.items()) if key != "apikey"
    )
    cache_path = os.path.join(CACHE_DIR, f"{current_date}_{params_str}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    response = requests.get(
        url="https://www.alphavantage.co/query?",
        params=typing.cast(typing.Mapping[str, typing.Any], params),
    )
    response.raise_for_status()
    data = response.json()
    if "Information" in data and "premium" in data["Information"]:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise ValueError("AlphaVantage API rate limit hit. Please try again later.")
    if "Error Message" in data:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise ValueError("Invalid call to AlphaVantage API.")
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data

def prices_daily(apikey: str, ticker: str, window: int = 252) -> pd.DataFrame:
    """
    Gets historical prices for a specified ticker from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        ticker (str): The stock ticker to get prices for
        window (int): The number of days to get prices for

    Returns:
        prices (pd.DataFrame): The historical prices
    """
    if window < 0:
        raise ValueError("window size must be positive")

    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "outputsize": "full",
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
    }

    json_data = get_alphavantage(params = params)
    full_time_series = list(json_data["Time Series (Daily)"].items())
    windowed_time_series = full_time_series[:window][::-1]
    prices_list = [
        [
            pd.to_datetime(key),
            float(value["1. open"]) * adj_ratio,
            float(value["5. adjusted close"]),
            float(value["2. high"]) * adj_ratio,
            float(value["3. low"]) * adj_ratio,
            float(value["1. open"]),
            float(value["4. close"]),
            float(value["2. high"]),
            float(value["3. low"]),
        ]
        for key, value in windowed_time_series
        if (adj_ratio := (float(value["5. adjusted close"]) / float(value["4. close"])))
    ]
    prices = pd.DataFrame(
        prices_list,
        columns=[
            "date",
            "adj_open",
            "adj_close",
            "adj_high",
            "adj_low",
            "open",
            "close",
            "high",
            "low",
        ],
    )
    prices.rename(columns={"adj_close": f"{ticker}_adj_close"}, inplace=True)

    if any(
        column not in prices.columns
        for column in ["date", "adj_open", f"{ticker}_adj_close", "adj_high", "adj_low"]
    ):
        raise ValueError("prices dataframe missing expected columns")
    return prices

def federal_funds_rates(apikey: str, window: int = 52 * 8) -> pd.DataFrame:
    """
    Gets the historical federal funds rates from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of months to get the federal funds rate for

    Returns:
        rates (pd.DataFrame): The historical federal funds rates
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "TREASURY_YIELD",
        "interval": "weekly",
        "maturity": "3month"
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "fed_funds"}, inplace=True)
    rates["fed_funds"] = pd.to_numeric(rates["fed_funds"])
    rates["fed_funds"] = rates["fed_funds"] / 100
    rates["fed_funds"] = (1 + rates["fed_funds"]) ** (1 /52) - 1

    if any(column not in rates.columns for column in ["date", "fed_funds"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

def treasury(apikey: str, window: int = 52 * 8) -> pd.DataFrame:
    """
    Gets the historical 3-month treasury yield from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of months to get the federal funds rate for

    Returns:
        rates (pd.DataFrame): The historical 3-month tresury yield
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "FEDERAL_FUNDS_RATE",
        "interval": "weekly",
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "3monhth_treasury_yield"}, inplace=True)
    rates["3monhth_treasury_yield"] = pd.to_numeric(rates["3monhth_treasury_yield"])
    rates["3monhth_treasury_yield"] = rates["3monhth_treasury_yield"] / 100

    if any(column not in rates.columns for column in ["date", "3monhth_treasury_yield"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

def unemployment(apikey: str, window: int = 12 * 8) -> pd.DataFrame:
    """
    Gets the historical unemployment rates from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of months to get the unemployment rate for

    Returns:
        rates (pd.DataFrame): The historical unemployment rates
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "UNEMPLOYMENT",
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "unemployment"}, inplace=True)
    rates["unemployment"] = pd.to_numeric(rates["unemployment"])
    rates["unemployment"] = rates["unemployment"] / 100

    if any(column not in rates.columns for column in ["date", "unemployment"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

def cpi(apikey: str, window: int = 12 * 8) -> pd.DataFrame:
    """
    Gets the historical CPI from the AlphaVantage API.

    Parameters:
        apikey (str): An AlphaVantage API key
        window (int): The number of years to get the inflation rates for

    Returns:
        rates (pd.DataFrame): The historical CPI
    """
    if window < 0:
        raise ValueError("window size must be positive")
        
    params: AVParameters = {
        "apikey": apikey,
        "datatype": "json",
        "function": "CPI",
    }

    json_data = get_alphavantage(params = params)
    windowed_time_series = json_data["data"][:window][::-1]
    
    rates = pd.DataFrame(windowed_time_series)
    rates["date"] = pd.to_datetime(rates["date"])
    rates.rename(columns={"value": "cpi"}, inplace=True)
    rates["cpi"] = pd.to_numeric(rates["cpi"])
    rates["cpi"] = rates["cpi"] / 100

    if any(column not in rates.columns for column in ["date", "cpi"]):
        raise ValueError("rates dataframe missing expected columns")
    return rates

if __name__ == "__main__":
    #key
    #alphavantage_api_key = os.getenv("av_api_key")
    alphavantage_api_key = "D32EVZCEU7HZUFYQ"
    
    #Time horizon
    years = 8
    
    #Monthly S&P 500 (SPY)
    spy = prices_daily(alphavantage_api_key, "SPY", 252 * years)[["date", "SPY_adj_close"]]
    spy["SPY_adj_close_1day"] = spy["SPY_adj_close"].shift(-1)
    spy["SPY_adj_close_1week"] = spy["SPY_adj_close"].shift(-5)
    spy["SPY_adj_close_1month"] = spy["SPY_adj_close"].shift(-20)
    spy["SPY_adj_close_3month"] = spy["SPY_adj_close"].shift(-60)
    spy["SPY_return_1day"] = (spy["SPY_adj_close_1day"] - spy["SPY_adj_close"]) / spy["SPY_adj_close"]
    spy["SPY_return_1week"] = (spy["SPY_adj_close_1week"] - spy["SPY_adj_close"]) / spy["SPY_adj_close"]
    spy["SPY_return_1month"] = (spy["SPY_adj_close_1month"] - spy["SPY_adj_close"]) / spy["SPY_adj_close"]
    spy["SPY_return_3month"] = (spy["SPY_adj_close_3month"] - spy["SPY_adj_close"]) / spy["SPY_adj_close"]
    spy = spy.drop(columns = ["SPY_adj_close_1week", "SPY_adj_close_1month", "SPY_adj_close_1day", "SPY_adj_close_3month"])
    spy.dropna(inplace = True)
    print(spy)
    
    #REIT ETF (SCHH)
    reit_tickers = ["EQR", "ESS", "AVB", "INVH"]
    
    reit_prices = {
        ticker: prices_daily(alphavantage_api_key, ticker, 252 * years)[["date", f"{ticker}_adj_close"]]
        for ticker in reit_tickers
    }
    
    #get 1week/1month/3month price
    for key, value in reit_prices.items():
        value[f"{key}_adj_close_1day"] = value[f"{key}_adj_close"].shift(-1)
        value[f"{key}_return_1day"] = (value[f"{key}_adj_close_1day"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        
        value[f"{key}_adj_close_1week"] = value[f"{key}_adj_close"].shift(-5)
        value[f"{key}_return_1week"] = (value[f"{key}_adj_close_1week"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        
        value[f"{key}_adj_close_1month"] = value[f"{key}_adj_close"].shift(-20)
        value[f"{key}_return_1month"] = (value[f"{key}_adj_close_1month"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        
        value[f"{key}_adj_close_3month"] = value[f"{key}_adj_close"].shift(-60)
        value[f"{key}_return_3month"] = (value[f"{key}_adj_close_3month"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        value = value.drop(columns = [f"{key}_adj_close_1day"])
        
        value[f"{key}_adj_close_1week"] = value[f"{key}_adj_close"].shift(-5)
        value[f"{key}_return_1week"] = (value[f"{key}_adj_close_1week"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        value = value.drop(columns = [f"{key}_adj_close_1week"])
        
        value[f"{key}_adj_close_1month"] = value[f"{key}_adj_close"].shift(-20)
        value[f"{key}_return_1month"] = (value[f"{key}_adj_close_1month"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        value = value.drop(columns = [f"{key}_adj_close_1month"])
        
        value[f"{key}_adj_close_3month"] = value[f"{key}_adj_close"].shift(-60)
        value[f"{key}_return_3month"] = (value[f"{key}_adj_close_3month"] - value[f"{key}_adj_close"]) / value[f"{key}_adj_close"]
        value = value.drop(columns = [f"{key}_adj_close_3month"])
    print(reit_prices)
    
    #merge
    df_reit = None
    for key, value in reit_prices.items():
        if df_reit is None:
            df_reit = value
        else:
            df_reit = df_reit.merge(value, on= "date", how = "outer")
    print(df_reit)
    
    #1-month Fed Funds Rate
    fed_funds = federal_funds_rates(alphavantage_api_key, 52 * years)
    print(fed_funds)
    
    #3-month treasury yield
    treasury = treasury(alphavantage_api_key, 52 * years)
    treasury["date"] = treasury["date"] + pd.Timedelta(days = 2)
    
    print(treasury)
    
    #Unemployment rate
    unemployment = unemployment(alphavantage_api_key, 12 * years)
    unemployment.set_index("date", inplace=True)
    unemployment = unemployment.resample("W").ffill()
    unemployment.index = unemployment.index + pd.Timedelta(days = 5)
    print(unemployment)
    
    #CPI
    cpi = cpi(alphavantage_api_key)
    cpi.set_index("date", inplace=True)
    cpi = cpi.resample("W").ffill()
    cpi.index = cpi.index + pd.Timedelta(days = 5)
    print(cpi)
    
    #merge
    df1 = pd.merge(spy, df_reit,
                     on = ["date"],
                     how = "left")
    
    df2 = pd.merge(unemployment, fed_funds,
                     on = ["date"],
                     how = "left")
    
    df2 = pd.merge(df2, treasury,
                     on = ["date"],
                     how = "left")
    print(df2)
    
    df1["date"] = pd.to_datetime(df1["date"])
    df2["date"] = pd.to_datetime(df2["date"])

    df = pd.merge(df1.sort_values("date"), df2.sort_values("date"), on="date")
    
    df = pd.merge(df, cpi,
                     on = ["date"],
                     how = "left")
    df.loc[df["date"].dt.year == 2024, "cpi"] = df.loc[df["date"].dt.year == 2024, "cpi"].ffill()
    df.dropna(inplace = True)
    print(df)
    
    #df.to_csv("/Users/jianihe/Desktop/us-residential-housing-return-prediction/data/av_data.csv")
    
    # Read the Redfin data into a DataFrame
    file_path_redfin = "/Users/jianihe/Desktop/weekly_housing_market_data_most_recent.tsv"
    df_redfin = pd.read_csv(file_path_redfin, sep="\t")
    
    #Extract the required features
    df_redfin.drop(columns=["region_type_id", "region_id"], inplace=True)

    #extract weekly data by county
    df_redfin = df_redfin[df_redfin["region_type"] == "county"]
    df_redfin = df_redfin[df_redfin["duration"] == "1 weeks"]
    df_redfin.drop(columns=["region_type", "duration"], inplace=True)
    df_redfin.dropna(inplace = True)

    #seperate state
    df_redfin["State"] = df_redfin["region_name"].str.split(", ").str[1]
    df_redfin["region_name"] = df_redfin["region_name"].str.split(", ").str[0]

    #define month
    df_redfin["period_begin"] = pd.to_datetime(df_redfin["period_begin"])
    
    

