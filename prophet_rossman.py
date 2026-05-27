from prophet import Prophet
from predictor import Predictor

REGRESSORS = [
    'Promo', 'DayOfWeek', 'Open', 'SchoolHoliday', 'StateHoliday', 'Day', 
    'Year', 'IsPromoMonth', 'DayOfWeek',
    'Month'
]

PROPHET_PARAMS = {
    'changepoint_prior_scale': 0.5, 
    'changepoint_range': 1.0, 
    'holidays_prior_scale': 0.3, 
    'seasonality_prior_scale': 0.01
}

class ProphetPredictor(Predictor):

    def predict_for_store(self, train_store, test_store):
        train_prophet = self._get_prophet_format(train_store)
        test_prophet = self._get_prophet_format(test_store)

        model = Prophet(**PROPHET_PARAMS)

        for regressor in REGRESSORS:
            model.add_regressor(regressor)

        model.fit(train_prophet)

        forecast = model.predict(test_prophet)
        prophet_preds = forecast.set_index("ds")["yhat"].clip(lower=0)

        test_store['Predicted_Sales'] = prophet_preds.values
        test_store[test_store['Open'] == 0]['Predicted_Sales'] = 0

        return test_store

    def _get_prophet_format(self, data):
        prophet_format = data.copy()
        prophet_format = prophet_format.rename(columns={'Date': 'ds'})

        if 'Sales' in prophet_format.columns:
            prophet_format = prophet_format.rename(columns={'Sales': 'y'})

        return prophet_format