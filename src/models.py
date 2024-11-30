from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
from .custom_logger import logger
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ModelHub:
    def __init__(self, data):
        self.data = data
        logger.info("Initialized ModelHub with the loaded dataset.")
        self.models = {
            'arima': self.setup_arima,
            'auto_arima': self.setup_auto_arima,
            'sarima': self.setup_sarima
        }
        self.arima = None
        self.auto_arima = None
        self.sarima = None
        self.model_dir = 'models'

    def setup_arima(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'arima_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up ARIMA model, this may take around 2 minutes...")
            self.arima = auto_arima(self.data['Production_Index'],
                                    start_p=1,
                                    start_d=1,
                                    start_q=1,
                                    max_p=6,
                                    max_d=3,
                                    max_q=6,
                                    seasonal=True,
                                    m=12,
                                    start_P=1,
                                    start_D=1,
                                    start_Q=1,
                                    max_P=4,
                                    max_D=3,
                                    max_Q=4,
                                    trend='ct',
                                    information_criterion='bic',
                                    trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
            joblib.dump(self.arima, model_path)
            logger.info("ARIMA model setup done and saved to disk.")
        else:
            logger.info("Loading ARIMA model from disk.")
            self.arima = joblib.load(model_path)

        if visualize:
            self.arima.plot_diagnostics(figsize=(11, 9))
            plt.savefig('visualizations/arima_diagnostics.png')
            logger.info("ARIMA diagnostics visualization saved to visualization/arima_diagnostics.png")

    def setup_auto_arima(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'auto_arima_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up Auto ARIMA model...")
            self.auto_arima = auto_arima(self.data['Production_Index'], seasonal=True, m=12)
            joblib.dump(self.auto_arima, model_path)
            logger.info("Auto ARIMA model setup done and saved to disk.")
        else:
            logger.info("Loading Auto ARIMA model from disk.")
            self.auto_arima = joblib.load(model_path)

        if visualize:
            self.auto_arima.plot_diagnostics(figsize=(11, 9))
            plt.savefig('visualizations/auto_arima_diagnostics.png')
            logger.info("Auto ARIMA diagnostics visualization saved to visualization/auto_arima_diagnostics.png")

    def setup_sarima(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'sarima_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up SARIMA model...")
            self.sarima = SARIMAX(self.data['Production_Index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            self.sarima = self.sarima.fit(disp=False)
            joblib.dump(self.sarima, model_path)
            logger.info("SARIMA model setup done and saved to disk.")
        else:
            logger.info("Loading SARIMA model from disk.")
            self.sarima = joblib.load(model_path)

        if visualize:
            self.sarima.plot_diagnostics(figsize=(11, 9))
            plt.savefig('visualizations/sarima_diagnostics.png')
            plt.close()
            logger.info("SARIMA diagnostics visualization saved to visualization/sarima_diagnostics.png")

            # Plot actual vs predicted values
            forecast = self.sarima.get_forecast(steps=len(self.data))
            predicted_mean = forecast.predicted_mean
            actual = self.data['Production_Index']

            plt.figure(figsize=(14, 7))
            plt.plot(actual, label='Actual')
            plt.plot(predicted_mean, label='Predicted', alpha=0.7)
            plt.legend()
            plt.title('Actual vs Predicted values')
            plt.savefig('visualizations/sarima_actual_vs_predicted.png')
            plt.close()
            logger.info("SARIMA actual vs predicted visualization saved to visualization/sarima_actual_vs_predicted.png")

    def init_model(self, model_name, visualize=False, force_retrain=False):
        if model_name in self.models:
            self.models[model_name](visualize, force_retrain)
        else:
            raise ValueError(f"Model {model_name} is not implemented.")

    def inference(self, model_name, n_periods):
        if model_name == 'arima' and self.arima is not None:
            logger.info(f"Making {n_periods} future predictions with ARIMA model.")
            forecast = self.arima.predict(n_periods=n_periods)
            logger.info(f"Done, {n_periods} periods ahead using ARIMA model.")
            return forecast
        elif model_name == 'auto_arima' and self.auto_arima is not None:
            logger.info(f"Making {n_periods} future predictions with Auto ARIMA model.")
            forecast = self.auto_arima.predict(n_periods=n_periods)
            logger.info(f"Done, {n_periods} periods ahead using Auto ARIMA model.")
            return forecast
        elif model_name == 'sarima' and self.sarima is not None:
            logger.info(f"Making {n_periods} future predictions with SARIMA model.")
            forecast = self.sarima.get_forecast(steps=n_periods).predicted_mean
            forecast = forecast.values  # Convert to numpy array
            logger.info(f"Done, {n_periods} periods ahead using SARIMA model.")
            return forecast
        else:
            raise ValueError(f"Model {model_name} is not initialized or not implemented.")
