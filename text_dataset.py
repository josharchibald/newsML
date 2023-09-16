''' text_dataset.py '''

''' Author: Joshua Archibald September 2023 '''

''' This file conatins the text_dataset class which contains methods 
    for the dataset for data loader '''

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class text_dataset(Dataset):
  
  ''' This method gets all the file names for all the inputs and labels for \
      the class. '''
  def __init__(self):
    
    self.text_files = 'news_inputs/text_data_input.h5'

    self.keywords_files = 'news_inputs/keywords_data_input.h5'

    self.stock_input_file = 'stock_inputs/stock_inputs.h5'

    self.stock_labels_file = 'stock_labels/stock_labels.h5'

    self.stock_profits_file = 'stock_profits/stock_profits.h5'

    self.sp500_profits_file = 'sp500_profits/sp500_profits.h5'
    
    self.text_file_names = []

    self.keywords_file_names = []

    with h5py.File(self.text_files, 'r') as f:
      for set in f:
        self.text_file_names.append(set)

    with h5py.File(self.keywords_files, 'r') as f:
      for set in f:
        self.keywords_file_names.append(set)

    with h5py.File(self.stock_input_file, 'r') as f:
      for set in f:
        self.stock_inputs = f[set][:]

    with h5py.File(self.stock_labels_file, 'r') as f:
      for set in f:
        self.stock_labels = f[set][:]

    with h5py.File(self.stock_profits_file, 'r') as f:
      for set in f:
        self.stock_profits = f[set][:]
    
    with h5py.File(self.sp500_profits_file, 'r') as f:
      for set in f:
        self.sp500_profits = f[set][:]

  def __len__(self):

    return len(self.text_file_names)
  
  def __getitem__(self, idx):

    text_file = self.text_file_names[idx]

    keywords_file = self.keywords_file_names[idx]

    stock_input = self.stock_inputs[idx]

    stock_label = self.stock_labels[idx]

    stock_profit = self.stock_profits[idx]

    sp500_profit = self.sp500_profits[idx]

    # Read Text Data
    with h5py.File(self.text_files, 'r') as f:
        text = f[text_file][:]

    # Read Keywords Data
    with h5py.File(self.keywords_files, 'r') as f:
        keywords = f[keywords_file][:]

    conv_features = np.stack((text, keywords), axis=0)

    conv_tensor = torch.tensor(conv_features, dtype=torch.float32)

    lin_tensor = torch.tensor(stock_input, dtype=torch.float32)

    labels_tensor = torch.tensor(stock_label, dtype=torch.float32)

    ''' the conv and lin tensor are for inputs to the ml model and the labels 
        are the labels and the stock_profit and sp500_profit are for model 
        performance calcs. '''
    return conv_tensor, lin_tensor, labels_tensor, stock_profit, sp500_profit
