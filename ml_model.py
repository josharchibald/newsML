''' ml_model.py '''

''' Author: Joshua Archibald September 2023 '''

''' This file conatins the ml_model class which contains methods 
    for the meachine learning model '''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ml_model(nn.Module):
  ''' This method sets up all the layers for the ml model '''
  def __init__(self, num_conv_layers, num_lin_layers, activation, \
                input_conv_channels, output_conv_channels, kernels, \
                input_lin_channels, output_lin_channels, pool_window, \
                sequence_length, merge_size, output_size, dropout_prob, \
                loss_function):
    
    super(ml_model, self).__init__()

    self.num_conv_layers = num_conv_layers
    self.num_lin_layers = num_lin_layers
    self.activation = activation
    self.input_conv_channels = input_conv_channels
    self.output_conv_channels = output_conv_channels
    self.kernels = kernels
    self.input_lin_channels = input_lin_channels
    self.output_lin_channels = output_lin_channels
    self.pool_window = pool_window
    self.sequence_length = sequence_length
    self.merge_size = merge_size
    self.output_size = output_size
    self.dropout_prob = dropout_prob
    self.loss_function = loss_function

    self.conv_layers = nn.ModuleList()
    self.lin_layers = nn.ModuleList()

    ''' Conv1D Branch '''
    self.conv_layers.append(nn.Conv1d(in_channels=self.input_conv_channels, \
                                    out_channels=self.output_conv_channels, \
                                    kernel_size=self.kernels))
    
    self.conv_layers.append(nn.BatchNorm1d(self.output_conv_channels))
    
    for _ in range(self.num_conv_layers - 1):

      self.conv_layers.append(nn.Conv1d(in_channels= \
                                      self.output_conv_channels, \
                                      out_channels=self.output_conv_channels, \
                                      kernel_size=self.kernels))
      
      self.conv_layers.append(nn.BatchNorm1d(self.output_conv_channels))

    self.conv_layers.append(nn.MaxPool1d(self.pool_window))

    self.conv_layers.append(nn.Dropout(self.dropout_prob))
      
    ''' Linear Branch '''
    self.lin_layers.append(nn.Linear(self.input_lin_channels, \
                                     self.output_lin_channels))
    
    self.lin_layers.append(nn.LayerNorm(self.output_lin_channels))

    for _ in range(self.num_lin_layers - 1):

      self.lin_layers.append(nn.Linear(self.output_lin_channels, \
                                       self.output_lin_channels))
      
      self.lin_layers.append(nn.LayerNorm(self.output_lin_channels))

    self.lin_layers.append(nn.Dropout(self.dropout_prob))

    ''' Merging of branches '''
    self.merge_dim = self.calculate_merge_dim(self.sequence_length)
    self.merge_layer = nn.Linear(self.merge_dim, self.merge_size) 

    ''' Output Layer ''' 
    self.output_layer = nn.Linear(self.merge_size, self.output_size) 


  ''' This method calculates the dimensions for the merging layer to merge 
      the convolutional layer and the linear layer. 
      
      Inputs: sequence_length (int), and other class variables used
      
      Outputs: merge_dim (int) '''
  def calculate_merge_dim(self, sequence_length):

    # Calculating Conv1D output size
    for _ in range(self.num_conv_layers):
        sequence_length = sequence_length - self.kernels + 1

    sequence_length = sequence_length // self.pool_window 

    # The Conv1D branch is then flattened
    conv_output_size = sequence_length * self.output_conv_channels

    # Linear branch output size is output_lin_channels
    lin_output_size = self.output_lin_channels

    # Merge dimension is sum of both
    merge_dim = conv_output_size + lin_output_size

    return merge_dim
  
  ''' These methods hold custom activation functions that can be used.
      
      Inputs: x (tensor)
      
      Outputs: tensor with activation function '''
  def swish(self, x):
    return x * torch.sigmoid(x)
  
  def mish(self, x):
    return x * torch.tanh(F.softplus(x))


  ''' This method is the forward pass for the model. It passes the inputs 
      through all the layers and parameterized activation functions, merges the
      convolution and linear branches and returns the multi-label classification
      output. 
      
      Inputs: x_conv, x_lin (tensors)
      
      Outputs: output (tensors)'''
  def forward(self, x_conv, x_lin):

    # Conv1D Branch
    for layer in self.conv_layers:

      if isinstance(layer, nn.Dropout):  # to handle dropout layer
        x_conv = layer(x_conv)

      if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.BatchNorm1d):
        x_conv = layer(x_conv)
        
        if self.activation == 'relu':
          x_conv = F.relu(x_conv)
        elif self.activation == 'tanh':
          x_conv = torch.tanh(x_conv)
        elif self.activation == 'prelu':
          x_conv = F.prelu(x_conv)
        elif self.activation == 'elu':
          x_conv = F.elu(x_conv)
        elif self.activation == 'swish':
          x_conv = self.swish(x_conv)
        elif self.activation == 'gelu':
          x_conv = F.gelu(x_conv)
        elif self.activation == 'mish':
          x_conv = self.mish(x_conv)
        else:
          raise ValueError("Invalid activation choice.")
          
      elif isinstance(layer, nn.MaxPool1d):
        x_conv = layer(x_conv)
    
    x_conv = x_conv.view(x_conv.size(0), -1)  # Flatten the output

    # Linear Branch
    for layer in self.lin_layers:

      if isinstance(layer, nn.Dropout):  # New lines to handle dropout layer
        x_lin = layer(x_lin)

      if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
        x_lin = layer(x_lin)
        
        if self.activation == 'relu':
          x_lin = F.relu(x_lin)
        elif self.activation == 'tanh':
          x_lin = torch.tanh(x_lin)
        elif self.activation == 'prelu':
          x_lin = F.prelu(x_lin)
        elif self.activation == 'elu':
          x_lin = F.elu(x_lin)
        elif self.activation == 'swish':
          x_lin = self.swish(x_lin)
        elif self.activation == 'gelu':
          x_lin = F.gelu(x_lin)
        elif self.activation == 'mish':
          x_lin = self.mish(x_lin)
        else:
          raise ValueError("Invalid activation choice.")

    # Merge the outputs of the Conv1D and Linear branches
    x_merged = torch.cat((x_conv, x_lin), dim=1)
    x_merged = self.merge_layer(x_merged)

    # Final output layer
    output = self.output_layer(x_merged)
  
    if self.loss_function == 'BCE_weighted':
      pass
    elif self.loss_function == 'BCE':
      output = torch.sigmoid(output) 
    else:
      raise ValueError("Invalid loss function choice.")
    
    return output