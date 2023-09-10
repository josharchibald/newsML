''' stock_data_processing.py '''

''' Author: Joshua Archibald September 2023 '''

import news_data_processing
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
import os
import h5py


class stock_data_processing():
    
  def __init__(self):

    ''' Cute message when object is made '''
    print('Starting Stock Data Processing') 
    print(80 *'-')   


  ''' This method gets and returns the stock symbols from the s&p 500

      Inputs: None

      Outputs: tickers (list) '''
  def fetch_sp500_list(self):
      
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    print('Fetching S&P500 Stock Symbols')

    tables = pd.read_html(url)

    sp500_table = tables[0]

    tickers = sp500_table['Symbol'].tolist()

    print('Success')

    return tickers
  
  ''' This method takes in the news_df that has the dates that the news is 
      captured and the tickers from the s&p 500 and returns only those stocks 
      that have prives at the beginign of when the news data set begins and
      are thus eligible to use, excluding those that may not have existed 
      through the entire training set time.
      
      Inputs: tickers (list), news_df (pd data frame)
      
      Outputs: usable_tickers (list) '''
  def exclude_invalid_tickers(self, tickers, news_df):

    usable_tickers = []

    for symbol in tickers:

      usable_tickers.append(symbol)

    for ticker in tqdm(tickers):

      ticker_obj = yf.Ticker(ticker)
      # using the second week start to avoid the holidays in January
      start_date = news_df['adjusted_date'].iloc[2]

      # Fetch historical data between start_date and latest data
      historical_data = ticker_obj.history(start=start_date)

      try:
        # see if the there is stock data from the start date near the start
        # of the dataset
        historical_data.loc[start_date, 'Open']
    
      except:
        # if the prices could not be found assume the stock doesn't yet exist
        # and remove it from the eligible list
        usable_tickers.remove(ticker)

    ''' this is to make sure not too many stocks where excluded '''
    print(f'Stocks remaining: {len(usable_tickers)}')

    return usable_tickers


  ''' This method takes in a list of stock symbols in a ticker list and 
      returns a dataframe with the symbol and its coresponding indusrty.

      Inputs: tickers (list)

      Outputs: industry_df (pd dataframe) '''
  def fetch_industry(self, tickers):
     
    stock_industry = []

    print('Fetching Industries')
    
    for ticker in tqdm(tickers):
      
      ticker_obj = yf.Ticker(f'{ticker}')

      stock_info = ticker_obj.info

      industry = stock_info.get('industry', 'N/A')

      stock_industry.append({'ticker': ticker, 'industry': industry})

    industry_df = pd.DataFrame(stock_industry)

    return industry_df
  
  ''' This method takes in a df with a ticker column and industry column and
      groups the tickers into list with common industries.
      
      Inputs: df (pd dataframe )
      
      Outputs: grouped_df (pd dataframe) '''
  def group_by_industry(self, df):

    grouped_df = df.groupby('industry')['ticker'].apply(list).reset_index()

    return grouped_df
  

  ''' This method takes in a start_date an end_date and a list of ticker 
      stock symbols and gets the historical opening price data for all the 
      stocks and combines it into a data frame which is then returned.
      
      Inputs: tickers (list), start_date, end_date (strs)
      
      Outputs: combined_data (pd data frame) '''
  def fetch_historical_stock_data(self, tickers, start_date, end_date):

    dfs = []

    buffer = 10

    print('Fetching historical stock data')

    for ticker in tqdm(tickers):

      ticker_obj = yf.Ticker(ticker)

      # Convert the string to a datetime object
      fetch_end_date = datetime.strptime(end_date, "%Y-%m-%d")
      # add buffer days to ensure we get the data
      fetch_end_date = fetch_end_date + timedelta(days=buffer)
      # Convert the datetime object back to a string
      fetch_end_date = fetch_end_date.strftime("%Y-%m-%d")

      # Convert the string to a datetime object
      fetch_start_date = datetime.strptime(start_date, "%Y-%m-%d")
      # subtrack buffer days to ensure we get the data
      fetch_start_date = fetch_start_date + timedelta(days=-buffer)
      # Convert the datetime object back to a string
      fetch_start_date = fetch_start_date.strftime("%Y-%m-%d")

      # Fetch historical data between start_date and end_date
      historical_data = ticker_obj.history(start=fetch_start_date, \
                                         end=fetch_end_date)
      
      # Extract the 'Open' prices and add to the combined_data DataFrame
      dfs.append(historical_data[['Open']].rename( \
                                                      columns={'Open': ticker}))
      combined_data = pd.concat(dfs, axis=1)

    return combined_data

  

  ''' This method takes in a ticker sybmbol a start_date and an end_date then 
      outputs the ratio of the price of the ticker at the end_date to the 
      price of the ticker at the start_date which is used as a metric for its 
      performance. 
      
      Inputs: ticker, start_date, end_date (str), combined_data (pd data frame)
      
      Outputs: ratio (float) '''
  def calulate_price_ratio(self, ticker, start_date, end_date, combined_data):

    ratio = 1

    try:
        # Get the opening prices on the start and end dates
        opening_price_start = combined_data.loc[start_date, ticker]
        opening_price_end = combined_data.loc[end_date, ticker]
        
        # Calculate the percentage increase
        ratio = opening_price_end / opening_price_start
    
    except:
        # if the prices could not be found then just return that it stayed the
        # same ie ratio of 1. Reasons for no price coulf be a holiday or 
        # disaster that caused markets to close. 
        ratio = 1

    return ratio

  ''' This method takes in the news_df and the stock_df and uses the dates in
      the news_df to find the performance between weeks for the industries in 
      the stock_df using the combined_data which will be used as inputs and then 
      shifted to be used as labels.
      
      Inputs: news_df, stock_df, combined_data (pd dataframe)
      
      Outputs: data_set_labels, past_industry_performance (list) '''
  def fetch_labels_industry_performance(self, news_df, stock_df, combined_data):

    stock_inputs_directory = 'stock_inputs'

    stock_labels_directory = 'stock_labels'

    if not os.path.exists(stock_inputs_directory):
      os.makedirs(stock_inputs_directory)

    if not os.path.exists(stock_labels_directory):
      os.makedirs(stock_labels_directory)
    
    ''' This is the list that will contain the performances for all the 
        weeks '''
    past_industry_performance = []

    print('Creating Labels')

    for date_index in tqdm(range(len(news_df['adjusted_date']) - 1)):
      ''' This list will contain the performances for each individual week '''
      weekly_performance = []

      start_date = news_df['adjusted_date'].iloc[date_index]
      end_date = news_df['adjusted_date'].iloc[date_index + 1]

      for industry in stock_df['ticker']:

        ''' For each stock in an industry 1 is added to the investment and then
            the ratio of the price at the start date and end date is added to 
            the income '''
        investment = 0
        income = 0

        for ticker in industry:
          
          investment += 1

          income += self.calulate_price_ratio(start_date=start_date, \
                                              end_date=end_date, \
                                              ticker=ticker, \
                                              combined_data=combined_data)
        
        if income > investment:
          ''' If the industry did will hence the inmcome was more than the 
              investment the performance is 1 otherise -1 for that industry '''
          weekly_performance.append(1)

        else:

          weekly_performance.append(-1)

      past_industry_performance.append(weekly_performance)
    
    ''' The labels are the performace of the industries shifted to the left 
        since he lables are how they perform in the future. '''
    data_set_labels = [past_industry_performance[1:] + \
                [past_industry_performance[len(past_industry_performance) - 1]]]
    
    data_set_labels = np.squeeze(np.array(data_set_labels))

    past_industry_performance = np.array(past_industry_performance)

    with h5py.File(f'{stock_inputs_directory}/stock_inputs.h5', 'w') as hf:
      # Create a dataset in the file
      hf.create_dataset(f'stock_data_inputs', \
                        data=past_industry_performance)
      
    with h5py.File(f'{stock_labels_directory}/stock_labels.h5', 'w') as hf:
      # Create a dataset in the file
      hf.create_dataset(f'stock_labels', \
                        data=data_set_labels)

    return data_set_labels, past_industry_performance

    
def main():

  # mondays (0) 1 week apart
  weekdays = 0
  weeks = 1

  start_year = 2019

  end_year = 2019

  jan1 = '01-01'

  dec31 = '12-31'

  news = news_data_processing.news_data_processing()

  text_df, keywords_df = news.collect_news_data(start_year=start_year, \
                                                end_year=end_year)

  grouped_text_data = news.group_by_weekday(df=text_df, weekday=weekdays, \
                                            weeks=weeks)
  
  grouped_keywords_data = news.group_by_weekday(df=keywords_df, \
                                                weekday=weekdays, weeks=weeks)
  
  stock = stock_data_processing()

  tickers = stock.fetch_sp500_list()

  used_tickers = stock.exclude_invalid_tickers(tickers, grouped_text_data)

  historical_data = stock.fetch_historical_stock_data(tickers=used_tickers, \
                    start_date= f'{start_year}-{jan1}', \
                    end_date=f'{end_year}-{dec31}')

  industry_df = stock.fetch_industry(used_tickers)

  grouped_stock_df = stock.group_by_industry(industry_df)

  performance_labels, performance_inputs = \
  stock.fetch_labels_industry_performance(stock_df=grouped_stock_df, \
                                          news_df=grouped_keywords_data, \
                                          combined_data=historical_data)

  print(np.shape(performance_labels))

  print(np.shape(performance_inputs))

if __name__ == '__main__':

  main()