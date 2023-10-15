import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandScheduler
import wandb
import text_dataset
import ml_model
import h5py
import news_data_processing
import stock_data_processing
import numpy as np
import os
import shutil

class model_training():

  def __init__(self):

    ''' Cute messege for training '''
    print('Training the Model')
    self.prev_config = None
    self.first_data = None
 
  ''' This method takes in the outputs for a batch and the stock profits and 
      the sp500 profits. It will turn the outputs which came from a sigmoid to
      binary then it will see which industries the model decided to purchase
      and how much the percent profit is for those and then compare it to the
      percent profit it would have been for just buying sp500 etfs. 

      Inputs:
      labels_batch: 2D array or tensor with shape [batch_size, x], binary 
      labels 0 or 1

      stock_profits_batch: 2D array or tensor with shape [batch_size, x], 
      corresponding stock profits
      
      sp_profit_batch: 1D array or tensor with shape [batch_size], 
      corresponding S&P 500 profits

      Returns:
      A list of final results for each batch, shape [batch_size]. '''
  def compute_profit_over_sp500(self, outputs_batch, stock_profits_batch, \
                                sp_profit_batch, threshold=0.5):

    final_results = []

    result_for_ones = []
    
    # Iterate through each item in the batch
    for i in range(len(outputs_batch)):
      outputs = outputs_batch[i]
      
      # Convert sigmoid outputs to binary labels based on threshold
      labels = [1 if output >= threshold else 0 for output in outputs]
      
      stock_profits = stock_profits_batch[i]
      sp_profit = sp_profit_batch[i]

      # Get indices where labels are 1
      indices_of_ones = [index for index, value in enumerate(labels) if \
                         value == 1]

      # Calculate the sum of corresponding stock_profits
      sum_stock_profits = sum([stock_profits[index] for index in \
                               indices_of_ones])

      # Calculate the result according to the formula
      # (sum_stock_profits - len(indices_of_ones)) / len(indices_of_ones)
      result_for_one = 0
      if len(indices_of_ones) != 0:
        result_for_one = (sum_stock_profits - len(indices_of_ones)) / \
        len(indices_of_ones)

      result_for_ones.append(result_for_one)
      
      # Subtract the S&P 500 profit
      final_result = result_for_one - sp_profit
      final_results.append(final_result)

    return final_results, result_for_ones


  ''' The Training loop '''
  def train_model(self, config):

    # first get the data using hyper parameters
    weekdays = config['weekdays']
    weeks = config['weeks']

    # fraction of all data that is used for the tokenizing vocabulary since it can
    # be very large on the order of ones of terabytes
    sample_number = config['sample_number']

    start_year = 2013

    end_year = 2013

    jan1 = '01-01'

    dec31 = '12-31'

    sp500_tickers = ['SPY', 'IVV', 'VOO']

    ''' Beginning Data Processing '''
    ''' Data process only if its the first time or the data processing config
        parameters change '''
    if (self.first_data is None or weekdays != self.prev_config['weekdays'] \
    or weeks != self.prev_config['weeks']) and True:
      
      news_input_dir = 'news_inputs'
      stock_input_dir = 'stock_inputs'
      stock_labels_dir = 'stock_labels'
      stock_profits_dir = 'stock_profits'
      sp500_profits_dir = 'sp500_profits'

      if os.path.exists(news_input_dir):
          shutil.rmtree(news_input_dir)

      if os.path.exists(stock_input_dir):
          shutil.rmtree(stock_input_dir)
      
      if os.path.exists(stock_labels_dir):
          shutil.rmtree(stock_labels_dir)

      if os.path.exists(stock_profits_dir):
          shutil.rmtree(stock_profits_dir)

      if os.path.exists(sp500_profits_dir):
          shutil.rmtree(sp500_profits_dir)
      
      print('Beginning Data Processing')

      news = news_data_processing.news_data_processing()

      text_df, keywords_df = news.collect_news_data(start_year=start_year, \
                                                    end_year=end_year)

      grouped_text_data = news.group_by_weekday(df=text_df, weekday=weekdays, \
                                                weeks=weeks)
      
      grouped_keywords_data = news.group_by_weekday(df=keywords_df, \
                                                    weekday=weekdays,\
                                                    weeks=weeks)
      
      news.tokenize(df=grouped_text_data, column_name='text', \
                    samples=sample_number)

      news.tokenize(df=grouped_keywords_data, \
                                      column_name='keywords', \
                                      samples=sample_number)
      
      stock = stock_data_processing.stock_data_processing()

      tickers = stock.fetch_sp500_list()

      used_tickers = stock.exclude_invalid_tickers(tickers, grouped_text_data)

      historical_data = stock.fetch_historical_stock_data(\
                        tickers=used_tickers, \
                        start_date= f'{start_year}-{jan1}', \
                        end_date=f'{end_year}-{dec31}')

      industry_df = stock.fetch_industry(used_tickers)

      grouped_stock_df = stock.group_by_industry(industry_df)

      stock.fetch_labels_industry_performance(stock_df=grouped_stock_df, \
                                              news_df=grouped_keywords_data, \
                                              combined_data=historical_data)
  
      stock.fetch_sp500_performance(news_df=grouped_text_data, \
                                sp500_tickers=sp500_tickers, \
                                    combined_data=historical_data)
    
    ''' End of Data Processing '''

    print('Data Processing Finished')


    ''' Instantiate the model and its components '''

    # custom dataset
    full_dataset = text_dataset.text_dataset()

    # split into training and validation
    train_size = int(1 - config['validation_fraction'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, \
                                              [train_size, val_size])

    train_loader = DataLoader(train_dataset, \
                            batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, \
                            batch_size=int(config["batch_size"]), shuffle=False)
    
    ''' These are the values for the model that depend on the data_set '''
    input_conv_channels = 2
 
    with h5py.File(r'C:\Users\joshu\newsML\news_inputs\text_data_input.h5',\
                  'r') as f:
      sequence_len = np.shape(f['text_data_0000'])[0]

    with h5py.File(r'C:\Users\joshu\newsML\stock_labels\stock_labels.h5',\
                  'r') as f:
      input_lin_channels = np.shape(f['stock_labels'])[1]
    
    model = ml_model.ml_model(
    num_conv_layers=config['num_conv_layers'],
    num_lin_layers=config['num_lin_layers'],
    activation=config['activation'],
    input_conv_channels=input_conv_channels,
    output_conv_channels=config['output_conv_channels'],
    kernels=config['kernels'],
    input_lin_channels=input_lin_channels,
    output_lin_channels=config['output_lin_channels'],
    pool_window=config['pool_window'],
    sequence_length=sequence_len,
    merge_size=config['merge_size'],
    output_size=input_lin_channels,
    dropout_prob=config['dropout_prob'],
    loss_function=config['loss_function']
    )

    stock_labels_file = r'C:\Users\joshu\newsML\stock_labels\stock_labels.h5'

    with h5py.File(stock_labels_file, 'r') as f:
      for set in f:
        stock_labels = f[set][:]

    # Calculate number of zeros and ones for each class
    num_zeros = (stock_labels == 0).sum(axis=0)
    num_ones = (stock_labels == 1).sum(axis=0)

    # Calculate total samples and weights for weighted BCEloss
    total_samples = num_zeros + num_ones
  
    weight_for_1 = (1 / num_ones) * (total_samples) / 2.0

    weight_for_1_tensor = torch.tensor(weight_for_1, dtype=torch.float32)

    # try bce with or without weights
    if config['loss_function'] == 'BCE_weighted':
      criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_for_1_tensor)
    elif config['loss_function'] == 'BCE':
      criterion = torch.nn.BCELoss()
    else:
      raise ValueError("Invalid loss function choice.")
    
    # Initialize optimizer
    if config['optimizer'] == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=config['lr'], \
                             weight_decay=config.get('weight_decay', 0))
    elif config['optimizer'] == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=config['lr'], \
                            momentum=config.get('momentum', 0), \
                            weight_decay=config.get('weight_decay', 0))
    elif config['optimizer'] == 'RMSprop':
      optimizer = optim.RMSprop(model.parameters(), lr=config['lr'], \
                                alpha=config.get('alpha', 0.99), \
                                weight_decay=config.get('weight_decay', 0))
    else:
        raise ValueError("Invalid optimizer choice.")
    
    ''' Everything instantiated '''

    ''' Now for actual training loop '''
    
    for epoch in range(config["epochs"]):

      model.train()

      train_loss = 0.0

      for batch in train_loader:

        conv_x, lin_x, label, *_ = batch
        optimizer.zero_grad()
        output = model(conv_x, lin_x)
        print(f' min and max off output tr {output.min()}, {output.max()}')
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), \
                                       max_norm=config['max_norm'])
        optimizer.step()

      print(f'Training Epoch: {epoch} Loss: {loss / len(train_loader)}')

      # Validation loop
      model.eval()

      val_loss = 0.0
      profit_over_sp = 0.0
      model_profit = 0.0

      with torch.no_grad():
          for batch in val_loader:

            conv_x, lin_x, label, stock_profit, sp_profit = batch
            output = model(conv_x, lin_x)
            print(f' min and max off output val {output.min()}, {output.max()}')
            loss = criterion(output, label)
            val_loss += loss.item()
            profit_sp500, profit_model = self.compute_profit_over_sp500(\
              sp_profit_batch=sp_profit, \
              stock_profits_batch=stock_profit, \
              outputs_batch=output, threshold=.5)
            profit_over_sp += sum(profit_sp500)
            model_profit += sum(profit_model)

      avg_val_loss = val_loss / len(val_loader)
      avg_profit_over_sp = profit_over_sp / len(val_loader)
      avg_model_profit = model_profit / len(val_loader)

      print(f'Validation Epoch: {epoch} Val Loss: {avg_val_loss} \n \
            Percent Profit From Model: {avg_model_profit} Percent Profit Over \
              S&P 500: {avg_profit_over_sp}')

      tune.report(val_loss=avg_val_loss)

      # Save a checkpoint after each epoch
      with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(model, path)

    self.prev_config = config
  

def main():

  ''' Put the api key and nothing else in a wb_api_key.txt file'''
  with open(r'C:\Users\joshu\newsML\wb_api_key.txt', 'r') as f:
    key = f.read().strip()

  wandb.login(key=key)

  wandb_init = {
  "project": "news_ml",
  "group": "experiment_group",
  "job_type": "train",
  }

  space = {
    "weekdays": tune.choice([0]),
    "weeks": tune.choice([1]),
    "sample_number": tune.choice([10]),
    "validation_fraction": tune.choice([.5]),
    "batch_size": tune.choice([4]),
    "num_conv_layers": tune.choice([1]),
    "num_lin_layers": tune.choice([1]),
    # 'relu', 'tanh', 'prelu', 'swish'
    "activation": tune.choice(['tanh']), 
    "output_conv_channels": tune.choice([1]),
    "kernels": tune.choice([4]),
    "output_lin_channels": tune.choice([2]),
    "pool_window": tune.choice([4]),
    "merge_size": tune.choice([3]),
    "dropout_prob": tune.choice([.2]),
    "loss_function": tune.choice(['BCE', 'BCE_weighted']),
    "optimizer": tune.choice(['Adam', 'SGD', 'RMSprop']),
    "max_norm": tune.choice([1.0]),
    "lr": tune.choice([1e-4]),
    "weight_decay": tune.choice([1e-2]),
    "momentum": tune.choice([.1]),
    "alpha": tune.choice([.3]),
    "epochs": tune.choice([10])
  }
  

  # HyperBand Scheduler
  scheduler = HyperBandScheduler(metric="val_loss", mode="min")
                                                                                                                                     
  training = model_training()

  analysis = tune.run(
      training.train_model,
      config=space,
      scheduler=scheduler,
      callbacks=[WandbLoggerCallback(                                     
          project=wandb_init["project"],
          group=wandb_init["group"],
          job_type=wandb_init["job_type"],
          sync_config=True  # This will sync the Ray Tune config to Wandb
      )]
  )
  
  best_config = analysis.best_config
  print("Best hyperparameters found were: ", best_config)

  # Find the trial with the best end validation loss
  best_trial = analysis.get_best_trial("val_loss", "min")
  best_checkpoint_dir = best_trial.checkpoint.value

  # Here you can directly load the entire model
  best_model = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))

  # Create the directory if it doesn't exist.
  if not os.path.exists('best_news_ml_model'):
    os.makedirs('best_news_ml_model')

  torch.save(best_model, 'best_news_ml_model/best_model.pth')


if __name__ == '__main__':

  main()