import requests
import os
import pandas as pd
import numpy as np
from .preprocessing import preprocess_data, split_data
from .utils import download_data

def load_data(train_start_date='1950-01-01', test_start_date='2018-01-01'):
    download_data()
    df = preprocess_data()
    return split_data(df, train_start_date, test_start_date)