# newsML
This currently contains the data processing scripts that produce data and labels
needed to train a machine learning algorithm that predicts stock performance
based on news articles. 

# Prerequisits 
You will need to install these dependences:
requests pandas numpy tqdm scikit-learn h5py yfinance torch

You will also need an API key for the New York Times archive API. Visit this
site:
https://towardsdatascience.com/collecting-data-from-the-new-york-times-over-any-period-of-time-3e365504004

And follow the instructions just for getting the key, then put the key, and just
the key, in a file called news_api_key.txt in the same directory as this file. 

# Input and Label Creation
Since this only contains the data processing right now, just run the 
main_data_processing.py script to create inputs and labels.

