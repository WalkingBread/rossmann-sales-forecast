import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from rossmann_data import load_and_proprocess_data
from prophet_rossman import ProphetPredictor
from sarima_rossmann import SarimaxPredictor

from eval import rmspe, rmse, r2
from statsmodels.tsa.stattools import adfuller

@st.cache_data
def get_data():
    return load_and_proprocess_data()

def correlation_matrix_plot(cols, train_data):
    corr = train_data[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation of Features')

    st.pyplot(fig)

def sales_by_variables_plot(train_data, *cols):
    col1, col2, col3, col4 = cols

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.barplot(x=f'{col1}',y="Sales", data=train_data, ax=axes[0, 0], palette='tab10')
    sns.barplot(x=f'{col2}',y="Sales", data=train_data, ax=axes[0, 1], palette='tab10')
    sns.barplot(x=f'{col3}',y="Sales", data=train_data, ax=axes[1, 0], palette='tab10')
    sns.barplot(x=f'{col4}',y="Sales", data=train_data, ax=axes[1, 1], palette='tab10')

    axes[0, 0].set_title(f'Sales in {col1}')
    axes[0, 1].set_title(f'Sales in {col2}')
    axes[1, 0].set_title(f'Sales in {col3}')
    axes[1, 1].set_title(f'Sales in {col4}')

    st.pyplot(fig)

def rolling_mean_and_std_plot(data):
    fig, axes = plt.subplots(1,1)
    fig.set_figheight(6)
    fig.set_figwidth(16)

    rolmean = data.rolling(window=30).mean()
    rolstd = data.rolling(window=30).std()

    axes.plot(data, color='blue', label='Original')
    axes.plot(rolmean, color='red', label='Rolling Mean')
    axes.plot(rolstd, color='black', label='Rolling Std')

    axes.legend(loc='best')
    axes.set_title(f'Rolling Mean & Standard Deviation')
    axes.set_xlabel('Date')
    axes.set_ylabel('Sales')

    st.pyplot(fig)


st.title("Rossmann Sales Forecaster")

train, test = get_data()

store_id = st.selectbox("Select Store ID", sorted(train['Store'].unique()))

train_store = train[train['Store'] == store_id].reset_index(drop=True)
test_store = test[test['Store'] == store_id].sort_values(by=['Date']).reset_index(drop=True)

predict_tab, eval_tab, eda_tab = st.tabs(["📊 Sales Forecast", "📈 Model Evaluation", "Data Exploration"])

with predict_tab:
    st.subheader('Sales History')

    fig = plt.figure(figsize=(16, 3))

    plt.plot(train_store['Date'], train_store["Sales"], label='Historical Train Sales', color='blue')
    plt.title(f"Store {store_id} Sales History")
    plt.legend(loc='best')

    st.pyplot(fig)

    model_choice = st.radio("Choose Model", ["Prophet", "SARIMAX"], key=1)

    if model_choice == 'Prophet':
        predictor = ProphetPredictor()
    elif model_choice == 'SARIMAX':
        predictor = SarimaxPredictor()

    if st.button("Run Prediction"):
        with st.spinner(f"Training {model_choice} for Store {store_id}..."):
            results = predictor.predict_for_store(train_store, test_store)

            st.subheader("Sales Forecast")

            fig, axes = plt.subplots(2, 1)
            fig.set_figheight(6)
            fig.set_figwidth(16)

            axes[0].plot(train_store['Date'], train_store['Sales'], label='Historical Sales', color='blue')
            axes[0].plot(test_store['Date'], test_store['Predicted_Sales'], label="Forecast", color='red')
            axes[0].set_title(f"Store {store_id} Sales History + Forecast")
            axes[0].legend(loc='best')

            axes[1].plot(test_store['Date'], test_store['Predicted_Sales'], label="Forecast", color='red')
            axes[1].set_title("Future Timeline Predictions (test.csv)")
            axes[1].legend(loc='best')
            st.pyplot(fig)

            st.table(results[['Date', 'Predicted_Sales']])

with eval_tab:
    model_choice = st.radio("Choose Model", ["Prophet", "SARIMAX"], key=2)

    if model_choice == "Prophet":
        predictor = ProphetPredictor()
    else:
        predictor = SarimaxPredictor()

    if st.button("Evaluate"):
        with st.spinner(f"Evaluating {model_choice} for Store {store_id}..."):
            train_size = int(len(train_store) * 0.8)
            train_data = train_store[:train_size].copy(deep=True)
            test_data = train_store[train_size:].copy(deep=True)

            test_data = predictor.predict_for_store(train_data, test_data)

            fig, axes = plt.subplots(2,1)
            fig.set_figheight(6)
            fig.set_figwidth(16)

            axes[0].plot(train_data['Date'], train_data["Sales"], label='Train', color='blue')
            axes[0].plot(test_data['Date'], test_data["Sales"], label='Test', color='orange')

            axes[0].plot(test_data['Date'], test_data['Predicted_Sales'], label="Prediction", color='red')
            axes[0].legend(loc='best')

            axes[1].plot(test_data['Date'], test_data["Sales"], label='Test', color='orange')
            axes[1].plot(test_data['Date'], test_data['Predicted_Sales'], label="Prediction",color='red')
            axes[1].legend(loc='best')
            st.pyplot(fig)

            rmspe_score = rmspe(test_data['Sales'], test_data['Predicted_Sales'])
            st.metric(label="RMSPE", value=f"{rmspe_score:.4f}")

            rmse_score = rmse(test_data['Sales'], test_data['Predicted_Sales'])
            st.metric(label="RMSE", value=f"{rmse_score:.4f}")

            r2_score = r2(test_data['Sales'], test_data['Predicted_Sales'])
            st.metric(label="R2", value=f"{r2_score:.4f}")

            comparison = test_data[['Date', 'Sales', 'Predicted_Sales']]
            comparison['Difference'] = comparison['Sales'] - comparison['Predicted_Sales']

            st.table(comparison.head(100))

with eda_tab:
    with st.spinner("Generating plots..."):
        st.header(f'Store {store_id}')

        st.subheader(f'Rolling Mean & Standard Deviation of Sales Over Time')
        rolling_mean_and_std_plot(train_store['Sales'])

        st.subheader('Stationarity (Adfuller test)')
        adf_result = adfuller(train_store['Sales'].dropna())
        
        st.metric(label="P-value", value=f"{adf_result[1]:.4f}")
        st.metric(label="Stationary", value=f"{adf_result[1] <= 0.05}")

        st.subheader(f'Rolling Mean & Standard Deviation of Log Sales Over Time')
        rolling_mean_and_std_plot(train_store['SalesLog'])

        st.subheader('Sales in Different Time Frames')
        sales_by_variables_plot(train_store, 'Year', 'Month', 'Day', 'DayOfWeek')

        st.subheader('Sales By Other Variables')
        sales_by_variables_plot(train_store, 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday')

        st.subheader('Feature Correlation Matrix')
        cols = ['Sales', 'Promo', 'SchoolHoliday', 'StateHoliday', 'Open', 'DayOfWeek', 'Month']
        correlation_matrix_plot(cols, train_store)

        st.header('All Stores')

        st.subheader('Feature Correlation Matrix')
        global_cols = [
            'Sales', 'Customers', 'Promo', 'Open', 
            'SchoolHoliday', 'CompetitionDistance', 
            'CompetitionOpen', 'DayOfWeek', 'Month'
        ]
        correlation_matrix_plot(global_cols, train)
