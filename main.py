import logging
from src.data_loader import load_data
from src.models import ModelHub
import pandas as pd
from src.benchmark import benchmark_models


def main():
    df_train,df_test = load_data()
    
    hub = ModelHub(df_train)
    
    hub.init_model('arima', visualize=True)
    logging.info("Initialized ARIMA model with visualization.")
    hub.init_model('prophet')
    logging.info("Initialized Prophet model.")
    hub.init_model('xgboost',)
    logging.info("Initialized XGBoost model.")
    hub.init_model('random_forest')
    logging.info("Initialized Random Forest model.")
    


    print(hub.inference('arima', n_periods=12))
    print(hub.inference('prophet', n_periods=12))
    print(hub.inference('xgboost', n_periods=12))
    print(hub.inference('random_forest', n_periods=12))

    benchmark_results = benchmark_models(hub, df_test)
    print(benchmark_results)

if __name__ == "__main__":
    main()