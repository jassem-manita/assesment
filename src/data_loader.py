import requests
import os
import pandas as pd
import numpy as np
from .preprocessing import preprocess_data, split_data
from .utils import download_data

def load_data():
    download_data()
    df= preprocess_data()
    return split_data(df)