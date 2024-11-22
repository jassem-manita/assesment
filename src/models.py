from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
from .custom_logger import logger
import joblib
import os
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class ModelHub:
    def __init__(self, data):
        self.data = data
        logger.info("Initialized ModelHub with the loaded dataset.")
        self.models = {
            'arima': self.setup_arima,
            'prophet': self.setup_prophet,
            'xgboost': self.setup_xgboost,
            'random_forest': self.setup_random_forest
        }
        self.arima = None
        self.prophet = None
        self.xgboost = None
        self.random_forest = None
        self.model_dir = 'models'

    def setup_arima(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'arima_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up ARIMA model, this may take around 2 minutes...")
            self.arima = auto_arima(self.data['Production_Index'],
                                    start_p=0,
                                    start_d=0,
                                    start_q=0,
                                    max_p=4,
                                    max_d=4,
                                    max_q=4,
                                    seasonal=True,
                                    m=12,
                                    start_P=0,
                                    start_D=0,
                                    start_Q=0,
                                    max_P=3,
                                    max_D=2,
                                    max_Q=3,
                                    trend='c',
                                    information_criterion='aic',
                                    trace=False,
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

    def init_model(self, model_name, visualize=False, force_retrain=False):
        if model_name in self.models:
            self.models[model_name](visualize, force_retrain)
        else:
            raise ValueError(f"Model {model_name} is not implemented.")

    def setup_prophet(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'prophet_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up Prophet model...")
            self.prophet = Prophet(
                changepoint_prior_scale=0.5,
                growth='linear',
                yearly_seasonality=True,
                seasonality_mode='multiplicative'
            )
            df_prophet = self.data.reset_index().rename(columns={'Date': 'ds', 'Production_Index': 'y'})
            self.prophet.fit(df_prophet[['ds', 'y']])
            joblib.dump(self.prophet, model_path)
            logger.info("Prophet model setup done and saved to disk.")
        else:
            logger.info("Loading Prophet model from disk.")
            self.prophet = joblib.load(model_path)

    def setup_xgboost(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up XGBoost model...")
            X = self.data.drop(columns=['Production_Index'])
            y = self.data['Production_Index']
            
            self.xgboost = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            self.xgboost.fit(X, y)
            joblib.dump(self.xgboost, model_path)
            logger.info("XGBoost model setup done and saved to disk.")
        else:
            logger.info("Loading XGBoost model from disk.")
            self.xgboost = joblib.load(model_path)

    def setup_random_forest(self, visualize=False, force_retrain=False):
        model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
        
        if not os.path.exists(model_path) or force_retrain:
            logger.info("Setting up Random Forest model...")
            X = self.data.drop(columns=['Production_Index'])
            y = self.data['Production_Index']
            
            self.random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
            self.random_forest.fit(X, y)
            joblib.dump(self.random_forest, model_path)
            logger.info("Random Forest model setup done and saved to disk.")
        else:
            logger.info("Loading Random Forest model from disk.")
            self.random_forest = joblib.load(model_path)

    def inference(self, model_name, n_periods):
        if model_name == 'arima' and self.arima is not None:
            logger.info(f"Making {n_periods} future predictions with ARIMA model.")
            forecast = self.arima.predict(n_periods=n_periods)
            logger.info(f"Done, {n_periods} periods ahead using ARIMA model.")
            return forecast
        elif model_name == 'prophet' and self.prophet is not None:
            logger.info(f"Making {n_periods} future predictions with Prophet model.")
            future_periods = self.prophet.make_future_dataframe(periods=n_periods, include_history=False)
            forecast = self.prophet.predict(future_periods)
            logger.info(f"Forecasted {n_periods} periods ahead using Prophet model.")
            return forecast['yhat'].values
        elif model_name == 'xgboost' and self.xgboost is not None:
            logger.info(f"Making {n_periods} future predictions with XGBoost model.")
            X_future = self.data.drop(columns=['Production_Index']).iloc[-n_periods:]
            forecast = self.xgboost.predict(X_future)
            logger.info(f"Forecasted {n_periods} periods ahead using XGBoost model.")
            return forecast
        elif model_name == 'random_forest' and self.random_forest is not None:
            logger.info(f"Making {n_periods} future predictions with Random Forest model.")
            X_future = self.data.drop(columns=['Production_Index']).iloc[-n_periods:]
            forecast = self.random_forest.predict(X_future)
            logger.info(f"Forecasted {n_periods} periods ahead using Random Forest model.")
            return forecast
        else:
            raise ValueError(f"Model {model_name} is not initialized or not implemented.")
        