''' main_data_processing.py '''

''' Author: Joshua Archibald September 2023 '''

import stock_data_processing
import news_data_processing
import numpy as np

def main():

  message_separator_length = 80

  print(message_separator_length *'-')

  print('\
         This script will demo the creation of inputs and labels for \n \
         \n \
         training a machine learning model that predicts the performance of \n \
         \n \
         stock industries within the S&P 500 based on their recent \n \
         \n \
         performance and recent news articles.')
  
  print(message_separator_length *'-')

  # mondays (0) 1 week apart
  weekdays = 0
  weeks = 1

  # fraction of all data that is used for the tokenizing vocabulary since it can
  # be very large on the order of ones of terabytes
  fraction = .2

  start_year = 2013

  end_year = 2013

  jan1 = '01-01'

  dec31 = '12-31'

  print(f'\
          The parameters that govern the creation of the inputs and the \n \
          \n \
          labels are as follows: \n \
          \n \
          weekdays: which is an integer from 0 to 6 that represents a day \n  \
          \n \
          of the week Monday to Sunday that governs which day of the week \n \
          \n \
          the input data will start and end, so basically which weekday the \n \
          \n \
          ml will make predictions on. \n \n \
          weeks: which is a positive integer that represents how many weeks \n \
          \n \
          between each data chunk, so basically how many weeks worth of \n \
          \n \
          data the ml has to predict with. \n \n \
          fraction: which is a float that represents what fraction of the \n \
          \n \
          total news data collected will be used in the historical vocab \n \
          \n \
          governing how many features there will be for each article for \n \
          \n \
          each week. \n \n \
          start_year and end_year: which are both integers that are the \n \
          \n \
          start and end year for the data that will be trained with. \n \n \
          These parameters are currently {weekdays}, {weeks}, {fraction}, \n \
          \n \
          {start_year}, and {end_year} respectively.')
  
  print(message_separator_length *'-')

  news = news_data_processing.news_data_processing()

  text_df, keywords_df = news.collect_news_data(start_year=start_year, \
                                                end_year=end_year)

  grouped_text_data = news.group_by_weekday(df=text_df, weekday=weekdays, \
                                            weeks=weeks)
  
  grouped_keywords_data = news.group_by_weekday(df=keywords_df, \
                                                weekday=weekdays, weeks=weeks)
  
  news.tokenize(df=grouped_text_data, column_name='text', fraction=fraction)

  news.tokenize(df=grouped_keywords_data, \
                                  column_name='keywords',fraction=fraction)
  
  print(f'\
          The news data has been processed. From this the important outputs \n \
          \n \
          are the text_inputs and keywords_inputs which are 3D np arrays. \n \
          \n \
          These are to be used as inputs to the ml. They are stored in the \n \
          \n \
          news inputs directory as HDF5 files due to their size. The  \n \
          \n \
          The number of data chunks that the ml can train on is printed \n \
          \n \
          above.')
  
  print(message_separator_length *'-')
  
  stock = stock_data_processing.stock_data_processing()

  tickers = stock.fetch_sp500_list()

  used_tickers = stock.exclude_invalid_tickers(tickers, grouped_text_data)

  historical_data = stock.fetch_historical_stock_data(tickers=used_tickers, \
                    start_date= f'{start_year}-{jan1}', \
                    end_date=f'{end_year}-{dec31}')

  industry_df = stock.fetch_industry(used_tickers)

  grouped_stock_df = stock.group_by_industry(industry_df)

  stock.fetch_labels_industry_performance(stock_df=grouped_stock_df, \
                                          news_df=grouped_keywords_data, \
                                          combined_data=historical_data)
  sp500_tickers = ['SPY', 'IVV', 'VOO']
  
  stock.fetch_sp500_performance(news_df=grouped_text_data, \
                                sp500_tickers=sp500_tickers, \
                                    combined_data=historical_data)
  
  print(f'\
          The stock data has been processed. From this the important \n \
          \n \
          outputs are the stock_inputs, stock_labels, used for inputs and \n \
          \n \
          and labels for the ml, and then the stock_profits and the sp500 \n \
          \n \
          profits which will be used for performance metrics. These are all \n \
          \n \
          stored in HDF5 files.')
  
  
  print(message_separator_length *'-')

  print(f'\
          In order to access the inputs, labels, etc you can find the HDF5 \n \
          \n \
          files in their appropriately named directories. Then in your \n \
          \n \
          scipt write : with h5py.File([file_name.h5], [r]) as hf: then \n \
          \n \
          you can use for file in hf: to interact with the data in each.')
  
  print(message_separator_length *'-')

  print('Thank you')


if __name__ == '__main__':

  main()