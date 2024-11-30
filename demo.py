import streamlit as st
import logging
from src.data_loader import load_data
from src.models import ModelHub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("Time Series Model Inference and Comparison")

    # Initialize session state for model, n_periods, and date ranges
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = 'arima'
    if "n_periods" not in st.session_state:
        st.session_state.n_periods = 12
    if "train_start_date" not in st.session_state:
        st.session_state.train_start_date = '1950-01-01'
    if "test_start_date" not in st.session_state:
        st.session_state.test_start_date = '2018-01-01'

    # Select model and periods with session state
    st.session_state.selected_model = st.selectbox(
        "Select Model for Inference", 
        ['arima', 'auto_arima', 'sarima'], 
        index=['arima', 'auto_arima', 'sarima'].index(st.session_state.selected_model)
    )
    st.session_state.n_periods = st.number_input(
        "Number of Next Predictions", 
        min_value=1, max_value=100, 
        value=st.session_state.n_periods
    )

    # Select training and testing intervals
    st.session_state.train_start_date = st.date_input(
        "Training Start Date", 
        value=pd.to_datetime(st.session_state.train_start_date)
    ).strftime('%Y-%m-%d')
    st.session_state.test_start_date = st.date_input(
        "Testing Start Date", 
        value=pd.to_datetime(st.session_state.test_start_date)
    ).strftime('%Y-%m-%d')

    # Load data with chosen intervals
    df_train, df_test = load_data(st.session_state.train_start_date, st.session_state.test_start_date)
    hub = ModelHub(df_train)

    # Display chosen intervals
    st.write(f"Training Interval: {st.session_state.train_start_date} to {st.session_state.test_start_date}")
    st.write(f"Testing Interval: {st.session_state.test_start_date} onwards")

    # Run inference
    if st.button("Run Inference"):
        hub.init_model(st.session_state.selected_model, visualize=True)
        logging.info(f"Initialized {st.session_state.selected_model} model with visualization.")
        predictions = hub.inference(st.session_state.selected_model, n_periods=st.session_state.n_periods)
        st.write(f"Predictions for {st.session_state.selected_model} model:")
        st.write(predictions)
        # Plot Comparison Results
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_test['Production_Index'].values[:min(st.session_state.n_periods,len(df_test))], label='Actual', color='black')
        ax.plot(predictions, label=st.session_state.selected_model)
        st.pyplot(fig)  

    # Compare models
    if st.button("Compare All Models"):
        comparison_results = {}
        for model in ['arima', 'auto_arima', 'sarima']:
            hub.init_model(model)
            logging.info(f"Initialized {model} model.")
            predictions = hub.inference(model, n_periods=st.session_state.n_periods)
            comparison_results[model] = predictions
        
        # Display comparison results in Streamlit
        st.write("Comparison Results:")
        st.write(comparison_results)

        # Plot Comparison Results
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot predictions from all models
        for model, predictions in comparison_results.items():
            ax.plot(predictions, label=model)
        ax.plot(df_test['Production_Index'].values[:min(st.session_state.n_periods,len(df_test))], label='Actual', color='black')
        ax.set_title('Comparison of Model Predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Predictions')
        ax.legend()
        st.pyplot(fig)  # Display plot in Streamlit

if __name__ == "__main__":
    main()
