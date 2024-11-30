import logging
from src.data_loader import load_data
from src.models import ModelHub
import pandas as pd
from src.benchmark import benchmark_models
import matplotlib.pyplot as plt

def main():
    df_train, df_test = load_data()
    logging.info(f"Training data: {df_train.head()}")
    logging.info(f"Test data: {df_test.head()}")
    
    hub = ModelHub(df_train)
    
    hub.init_model('arima', visualize=True)
    logging.info("Initialized ARIMA model with visualization.")
    hub.init_model('auto_arima', visualize=True)
    logging.info("Initialized Auto ARIMA model with visualization.")
    hub.init_model('sarima', visualize=True)
    logging.info("Initialized SARIMA model with visualization.")
    
    arima_results = hub.inference('arima', n_periods=12)
    logging.info(f"ARIMA results: {arima_results}")
    auto_arima_results = hub.inference('auto_arima', n_periods=12)
    logging.info(f"Auto ARIMA results: {auto_arima_results}")
    sarima_results = hub.inference('sarima', n_periods=12)
    logging.info(f"SARIMA results: {sarima_results}")

    # Visualize SARIMA results
    plt.figure(figsize=(10, 6))
    plt.plot(df_train.index, df_train, label='Training Data')
    plt.plot(pd.date_range(start=df_train.index[-1], periods=12, freq='M'), sarima_results, label='SARIMA Forecast')
    plt.legend()
    plt.title('SARIMA Model Forecast')
    plt.show()

    print(arima_results)
    print(auto_arima_results)
    print(sarima_results)

    benchmark_results = benchmark_models(hub, df_test)
    logging.info(f"Benchmark results: {benchmark_results}")
    print(benchmark_results)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
