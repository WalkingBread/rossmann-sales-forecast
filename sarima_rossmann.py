from statsmodels.tsa.statespace.sarimax import SARIMAX
from predictor import Predictor

class SarimaxPredictor(Predictor):
    
    def predict_for_store(self, train_store, test_store):
        SARIMAX_model = SARIMAX(
            train_store[["Sales"]], 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 1, 7),
            exog=train_store[['Promo']]
        )
        SARIMAX_model_fit = SARIMAX_model.fit()

        forecast_steps = len(test_store)

        preds = SARIMAX_model_fit.forecast(forecast_steps, exog=test_store[['Promo']])

        test_store['Predicted_Sales'] = preds.values
        test_store['Open'] = test_store['Open'].fillna(1)
        test_store.loc[test_store['Open'] == 0, 'Predicted_Sales'] = 0

        return test_store