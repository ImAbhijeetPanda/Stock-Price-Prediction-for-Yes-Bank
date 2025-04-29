import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARIMAX Stock Forecast", layout="centered")

st.title("📈 ARIMAX Stock Forecasting App")
st.markdown("Forecast the stock closing price for the next *N* months using ARIMAX model.")

# Sidebar input
n_steps = st.sidebar.slider("Select number of months to forecast", 1, 12, 3)

@st.cache_resource
def load_model_and_data():
    model_fit = SARIMAXResults.load("arimax_model.pkl")
    df = pd.read_csv("processed_data.csv", index_col='Date')
    df.index = pd.to_datetime(df.index)
    return model_fit, df

model_fit, df = load_model_and_data()

def forecast_future_close_prices(n_steps=3):
    try:
        df_filtered = df[['Close', 'MA_2', 'MA_3', 'Momentum_1']].dropna()
        X = df_filtered[['MA_2', 'MA_3', 'Momentum_1']]
        y = df_filtered['Close']
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        # Simulate future exog with noise
        last_exog = X_train.iloc[-1]
        X_future = pd.DataFrame(
            [last_exog.values + np.random.normal(0, 0.01, size=len(last_exog)) for _ in range(n_steps)],
            columns=X_train.columns
        )

        # Predict future values
        y_future_pred = model_fit.predict(
            start=len(y_train),
            end=len(y_train) + n_steps - 1,
            exog=X_future
        ).round(2)

        # Create future dates
        last_date = y_test.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_steps, freq='M')

        forecast_df = pd.DataFrame({
            'Month': future_dates.strftime('%B %Y'),
            'Predicted Closing Price (₹)': y_future_pred.values
        })

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_train.index, y_train, label='Training Data', color='blue')
        ax.plot(y_test.index, y_test, label='Test Data', color='green')
        ax.plot(future_dates, y_future_pred, label='Future Forecast', color='red', linestyle='--', marker='o')
        ax.set_title(f'Forecast for Next {n_steps} Month(s)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.grid(True)
        ax.legend()

        return forecast_df, fig

    except Exception as e:
        st.error(f"❌ Forecasting failed: {str(e)}")
        return None, None

if st.button("🔮 Forecast Now"):
    forecast_df, fig = forecast_future_close_prices(n_steps)
    if forecast_df is not None:
        st.success("✅ Forecast completed!")
        st.dataframe(forecast_df)
        st.pyplot(fig)

