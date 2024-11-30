import pandas as pd
import numpy as np
from .custom_logger import logger

def preprocess_data(file_path='data/data.csv'):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'IPG2211A2N': 'Production_Index', 'DATE': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df = df.set_index('Date')
    df = df.asfreq('MS', fill_value=np.nan)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    logger.info('Data preprocessing done')
    return df

def split_data(df, train_start_date='1950-01-01', test_start_date='2018-01-01'):
    df_train = df.loc[train_start_date:test_start_date]
    df_test = df.loc[test_start_date:]
    logger.info('Train dataset size: %d', len(df_train))
    logger.info('Test dataset size: %d', len(df_test))
    return df_train, df_test





