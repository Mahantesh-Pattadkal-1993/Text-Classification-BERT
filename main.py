import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas
import torch
from sklearn.model_selection import train_test_split

# Setting up GPU/CPU for torch
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def trainset_creation():

     # Load data and set labels, complaints are marked as 0 and non complaints as 1
    data_complaint = pd.read_csv('data/complaint1700.csv')
    data_complaint['label'] = 0
    data_non_complaint = pd.read_csv('data/noncomplaint1700.csv')
    data_non_complaint['label'] = 1

    # Concatenate complaining and non-complaining data to create full data
    data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)

    # Drop 'airline' column
    data.drop(['airline'], inplace=True, axis=1)

    X = data.tweet.values
    y = data.label.values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)
    return X_train, X_val, y_train, y_val

def load_testset():
    # Load test data
    test_data = pd.read_csv('data/test_data.csv')

    # Keep important columns
    test_data = test_data[['id', 'tweet']]
    return test_data





