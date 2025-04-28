# 📈 Stock Price Prediction for Yes Bank

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📝 Project Overview
This project predicts the **monthly closing stock prices** of **Yes Bank** using Machine Learning (ML) and Time Series models. It focuses especially on the volatile period after 2018, following a major financial scandal involving the bank's leadership.

**Key Points:**
- Predict monthly stock closing prices.
- Compare ML models (Linear Regression, Ridge, Lasso, Random Forest, KNN, SVM) and Time Series models (ARIMA, SARIMAX/ARIMAX).
- Analyze impact of internal fraud (2018–2020) on stock price movements.

---

## 🏦 Business Context
- **Yes Bank**, a major Indian private bank, suffered a major crisis due to fraud in 2018.
- Stock prices crashed sharply post-2018, causing huge volatility.
- **Objective**: Build predictive models that can forecast stock prices accurately during and after crisis periods.

---

## 📂 Dataset Overview
- **Data Source**: Historic monthly stock prices (Open, High, Low, Close) of Yes Bank from 2005 to 2020.
- **Rows**: 185
- **Features**:  
  - `Date` (Month-Year)
  - `Open` (Opening Price)
  - `High` (Highest Price)
  - `Low` (Lowest Price)
  - `Close` (Closing Price — **Target**)

✅ No missing values or duplicates were found.

---

## 🛠 Project Workflow

1. **Data Preprocessing**  
   - Converted `Date` to `YYYY-MM-DD` format.
   - Set `Date` as the index.
   - Created new features: 2-month and 3-month moving averages, momentum feature.

2. **Exploratory Data Analysis (EDA)**  
   - Line plot of closing prices (2005–2020).
   - Highlighted 2018-2020 fraud period.
   - Candlestick chart.
   - Correlation heatmaps and pairplots.

3. **Feature Engineering**  
   - Rolling moving averages (MA2, MA3).
   - Momentum indicators.
   
4. **Model Building**
   - **ARIMA (1,0,1)**
   - **Common ML Models**:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Random Forest
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
   - **ARIMAX (SARIMAX)** with exogenous features.

5. **Evaluation Metrics**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score (for ML models)

---

## 📊 Model Performance Summary

| Model                   | RMSE   | MAE   | R² Score  |
|:------------------------|:-------|:------|:----------|
| Linear Regression       | 3.02   | 2.05  | 0.74      |
| Ridge Regression        | 4.20   | 3.30  | 0.50      |
| Lasso Regression        | 3.34   | 2.36  | 0.68      |
| Random Forest           | 12.23  | 10.62 | -3.28     |
| KNN                     | 20.62  | 19.75 | -11.17    |
| SVM                     | 21.06  | 19.44 | -11.69    |
| ARIMA(1,0,1)             | 48.77  | 44.37 | -         |
| **ARIMAX(3,0,3)**        | **1.29** | **1.16** | -      |

> 🏆 **Best Model**: **ARIMAX(3,0,3)** with RMSE = 1.29 and MAE = 1.16

---

## 🔮 Future Predictions
Using the ARIMAX model, the stock price for the next 3 months was forecasted:

| Month       | Predicted Closing Price (₹) |
|:------------|:----------------------------|
| May 2020    | 22.04                        |
| June 2020   | 20.19                        |
| July 2020   | 19.99                        |

---

## 🚀 Key Learnings
- **ARIMAX** models with external features outperform pure ML and ARIMA models for financial time series with external shocks.
- **Feature Engineering** (like moving averages, momentum) greatly enhances prediction performance.
- **Stationarity Testing** (ADF Test) and **AutoCorrelation Analysis** (ACF/PACF) are essential before applying time series models.

---

## 📢 Recommendations
- **Use ARIMAX** for short-term stock price predictions.
- **Continuously update** the model with fresh data for maintaining accuracy.
- **Incorporate more exogenous variables** like macroeconomic indicators (GDP, interest rates) or news sentiment for even better forecasting.

---

## 📚 Tech Stack
- Python 🐍
- Libraries:  
  - `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly`
  - `statsmodels` (ARIMA, SARIMAX)
  - `scikit-learn` (ML Models, Preprocessing)

---

## 🔗 Project Links
- [📂 GitHub Repository](https://github.com/ImAbhijeetPanda/Stock-Price-Prediction-for-Yes-Bank)
- [📒 Notebook on Colab](https://colab.research.google.com/github/ImAbhijeetPanda/Stock-Price-Prediction-for-Yes-Bank/blob/main/Stock_Price_Prediction_for_Yes_Bank.ipynb)

---

## 👨‍💻 Author
- **Name**: Abhijeet Panda
- **Contribution**: Individual Project
- **Project Type**: Regression

---
## Contact

For any questions or feedback, feel free to reach out:

- **Email**: [iamabhijeetpanda@gmail.com](mailto:iamabhijeetpanda@gmail.com)
- **LinkedIn**: [Abhijeet Panda](https://www.linkedin.com/in/imabhijeetpanda)
- **GitHub**: [ImAbhijeetPanda](https://github.com/ImAbhijeetPanda)
---

# ✅ Thank you for visiting the project!  
---
> *If you like it, give the repository a ⭐️!*
