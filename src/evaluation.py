import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from .custom_logger import logger

def eval_metrics(pred, actual):
    MAPE = round(mean_absolute_percentage_error(actual, pred), 4)
    MAE = round(mean_absolute_error(actual, pred), 3)
    RMSE = round(np.sqrt(mean_squared_error(actual, pred)), 3)
    
    logger.info('MAPE: '+str(MAPE * 100)+ '%  MAE: '+ str(MAE)+ '  RMSE: '+str(RMSE))
    return MAPE, MAE, RMSE


