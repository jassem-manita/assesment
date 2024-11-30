from .evaluation import eval_metrics
from .custom_logger import logging
import matplotlib.pyplot as plt

def benchmark_models(model_hub, df_test, visualize=True):
    results = {}
    models = ['arima', 'auto_arima', 'sarima']
    
    for model in models:
        predictions = model_hub.inference(model, n_periods=len(df_test))
        metrics = eval_metrics(predictions, df_test['Production_Index'].values)
        results[model] = {'metrics': metrics, 'predictions': predictions}
        logging.info(f"Benchmarked {model} model: {metrics}")
    
    if visualize:
        plt.clf()
        plt.figure(figsize=(14, 7))
        plt.plot(df_test['Production_Index'].values, label='Actual', color='black')
        
        for model, metrics in results.items():
            predictions = metrics['predictions']
            plt.plot(predictions, label=model)
        
        plt.legend()
        plt.title('Model Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Production Index')
        plt.savefig('visualizations/benchmark.png')
    return results
