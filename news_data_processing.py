''' news_data_processing.py '''

''' Author: Joshua Archibald September 2023 '''

''' This file conatins the news_data_processing class which contains methods 
    that collect and group  '''

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import h5py
import os
import sys
import random

class news_data_processing():
    
  def __init__(self):
      self.max_tokens_global = 0
      
      ''' Cute message when object is made '''
      print('Starting News Data Processing')
      print(80 *'-')
  
  ''' This method takes in a month and a year and returns the news articles
      and their information for that month and year.
      
      Inputs: month, year (ints)
      
      Outputs: news_data (json) '''
  def fetch_news_and_date(self, year, month):

    ''' Put the api key and nothing else in a news_api_key.txt file'''
    with open(r'C:\Users\joshu\newsML\news_api_key.txt', 'r') as f:
      key = f.read().strip()

    base_url = 'https://api.nytimes.com/svc/archive/v1/'
    url = base_url + '/' + str(year) + '/' + str(month) + '.json?api-key=' + key
    
    ''' I chose a short wait time and many tries so it gets the data asap 
        because for some reason even with the rate limit every like 7 requests
        it stalls for a bit '''
    max_retries = 12
    retries = 0
    retry_wait = 3
    
    while retries < max_retries:
      try:
        news_data = requests.get(url).json()

        # Check if the 'response' key exists in news_data
        if 'response' in news_data:
            return news_data
    
      except Exception as e:
        print(f'An error occurred: {e}, retrying...')
      
      retries += 1
      time.sleep(retry_wait)
        
    print('Max retries reached, exiting.')
    sys.exit()
  

  ''' This method takes in a start_year and an end_year, it gets the news data
      from every month between those years and returns  a data frame that has
       the keywords and date and then the headline and lead paragraph and
      date for all the headlines between those years.
      
      Inputs: start_year, end_year (ints)
      
      Outputs: text_df, keywords_df (pandas data frames) '''
  def collect_news_data(self, start_year, end_year):
    # Initialize lists to collect news text and keywords
    news_text = []
    news_keywords = []

    # Fixed length for date strings
    date_length = 10

    # There's a request rate limit 10 a min, but its iffy so theres a retry 
    # count as well
    request_wait = 6

    # Months of the year to get from
    months = 12

    # Goes through the start year to end year one month at a time
    for year in range(start_year, end_year + 1):

      print(f'Fetching news data for {year}')

      for month in tqdm(range(1, months + 1)):

        # Fetch news and date for the specific year and month
        article_data = self.fetch_news_and_date(year=year, month=month)

        # Wait due to the request rate limit
        time.sleep(request_wait)

        # Capture the headline and lead paragraph, date, and keywords for each
        # article in that month
        for obj in article_data['response']['docs']:
          
          # Extract the date and its year part
          date = obj['pub_date'][:date_length]
          # Extract the year part of the date
          date_year = int(date.split('-')[0])
          
          # Check if the year falls within the specified range
          if start_year <= date_year <= end_year:
            
            # Construct the text and keywords strings
            text = '{} {}'.format(obj['headline']['main'], \
                                  obj['lead_paragraph'])
            keywords = ''

            for keyword in obj['keywords']:
              keywords += '{} '.format(keyword['value'])

            # Append the text and date to the list
            news_text.append({'text': text, 'date': pd.Timestamp(date)})

            # Append the keywords and date to the list
            news_keywords.append({'keywords': keywords, 'date': \
                                  pd.Timestamp(date)})

    # Convert the lists to DataFrames
    text_df = pd.DataFrame(news_text)
    keywords_df = pd.DataFrame(news_keywords)

    return text_df, keywords_df


  ''' This method takes in a df with a dates column as well as a weekday and  
          a number of weeks and groups the data into lists, where each list 
          contains all the data that is the input number of weeks between the 
          input weekday. 

          Input: df (pandas data frame), weekday 0 for monday, 1 for tuesday, 2 
          for wednesday, ... 6 for sunday (int), weeks (int)

          Output: grouped_df which is the df with an adjusted_date column 
                  instead of the date column with the data put into lists
                  depending on which set of weeks they belong to '''
  def group_by_weekday(self, df, weekday, weeks):
    # Convert integer to its corresponding weekday label
    weekday_label = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][weekday]
    
    # Shift the dates so that the desired weekday becomes the start of the week
    df['adjusted_date'] = \
    df['date'].apply(lambda x: x - pd.Timedelta(days=(x.weekday() \
                                                      - weekday) % 7))
    
    # Resample by week, starting on the desired weekday
    grouped_df = \
    df.resample(f"{weeks}W-{weekday_label}", on='adjusted_date').apply(list)
    
    # Reset the index and convert dates back to string format for readability
    grouped_df = grouped_df.reset_index()
    grouped_df['adjusted_date'] = \
    grouped_df['adjusted_date'].dt.strftime('%Y-%m-%d')

    # Drop the old date column
    if 'date' in grouped_df.columns:
        grouped_df = grouped_df.drop(['date'], axis=1)
    
    return grouped_df
  
  ''' This method takes in a df and a column_name for the column the data is
      held and outputs the text data tokenized to zero padded 2D array inputs 
      for each week. This method takes in a number so that a random subset of 
      size samples is used for the historical vocab to limit the features per 
      news article since if its not limited the inputs can easily fill up a
      computers disk storage.
      
      Inputs: df (pd data frame), column_name (str), samples (int)
      
      Outputs: inputs_array (np array) '''
  def tokenize(self, df, column_name, samples):

    N = df[column_name].apply(len).max()

    input_dir = r'C:\Users\joshu\newsML\news_inputs'

    if not os.path.exists(input_dir):
      os.makedirs(input_dir)

    # Create an HDF5 file
    with h5py.File(f'{input_dir}/{column_name}_data_input.h5', 'w'):
      pass  # Doing nothing, just creating the file

    data_chunks = 0

    print(f'Tokenizing inputs for {column_name}')
    if column_name == 'text':
      print('Finding max tokens')
      # First Pass - Determine max number of tokens across all weeks
      for week_list in tqdm(df[column_name].iloc[:-1]):
        # Each week, randomly sample a subset of articles from that week to form
        # the historical corpus
        week_sample = random.sample(week_list, min(samples, len(week_list)))
        
        # Fit a TfidfVectorizer on the week's sample to get the vocabulary for 
        # this week
        weekly_vectorizer = TfidfVectorizer()
        weekly_vectorizer.fit(week_sample)
        fixed_vocab = weekly_vectorizer.get_feature_names_out()

        # Use the determined vocabulary to tokenize the entire week's data
        weekly_vectorizer = TfidfVectorizer(vocabulary=fixed_vocab)
        X = weekly_vectorizer.fit_transform(week_list)
        max_tokens_this_week = X.shape[1]  # Just look at the number of columns
        self.max_tokens_global = max(self.max_tokens_global, \
                                    max_tokens_this_week)
    print(f'Max tokens: {self.max_tokens_global}')
    print(f'Max number of articles a week {N}')
    print('Tokenizing')
    for week_list in tqdm(df[column_name].iloc[:-1]):

      # Tokenize the entire week's data using max_features
      weekly_vectorizer = TfidfVectorizer(max_features=self.max_tokens_global)
      X = weekly_vectorizer.fit_transform(week_list)
      X_dense = X.toarray()

      # Zero pad the columns to ensure all articles have max_tokens_global 
      # features
      padded_X_dense = np.zeros((X_dense.shape[0], self.max_tokens_global))
      padded_X_dense[:, :X_dense.shape[1]] = X_dense

      X_dense = padded_X_dense

      # Create and fit the scaler
      scaler = StandardScaler().fit(X_dense)

      # Apply the standard scaling
      X_dense = scaler.transform(X_dense)

      # Get the actual number of headlines for the current week
      n = X_dense.shape[0]

      # Zero pad to match the shape (N, num_features)
      if n < N:
          # Create a zero matrix of shape (N-n, num_features)
          zero_padding = np.zeros((N - n, X_dense.shape[1]))
          
          # Concatenate the actual data and the zero padding
          X_padded = np.vstack([X_dense, zero_padding])
      else:
          X_padded = X_dense

      X_flattened = X_padded.flatten()

      with h5py.File(f'{input_dir}/{column_name}_data_input.h5', 'a') as hf:
        # Create a dataset in the file
        hf.create_dataset(f'{column_name}_data_{data_chunks:04}', \
                          data=X_flattened)
        
      data_chunks += 1

    print(f'The number of data inputs is {data_chunks}')


''' Main for testing '''
def main():

  # mondays 1 week apart
  weekdays = 0
  weeks = 1

  samples = 50

  start_year = 2017

  end_year = 2017

  news = news_data_processing()

  text_df, keywords_df = news.collect_news_data(start_year=start_year, \
                                                end_year=end_year)

  grouped_text_data = news.group_by_weekday(df=text_df, weekday=weekdays, \
                                            weeks=weeks)
  
  grouped_keywords_data = news.group_by_weekday(df=keywords_df, \
                                                weekday=weekdays, weeks=weeks)
  
  news.tokenize(df=grouped_text_data, column_name='text', samples=samples)

  news.tokenize(df=grouped_keywords_data, \
                                  column_name='keywords', samples=samples)

  
if __name__ == '__main__':

  main()


