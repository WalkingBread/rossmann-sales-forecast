import pandas as pd
import numpy as np

DATA_DIR = 'rossmann-store-sales'

def load_and_proprocess_data():
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')

    stores = pd.read_csv(f'{DATA_DIR}/store.csv')

    train_merged = pd.merge(train, stores, on='Store', how='left')
    test_merged = pd.merge(test, stores, on='Store', how='left')

    train_merged = clean_data(train_merged)
    test_merged = clean_data(test_merged)

    return train_merged, test_merged


def is_promo_month(row):
    if row['PromoInterval'] != 0:
        month_name = row['Date'].strftime('%b')
        
        if month_name in row['PromoInterval']:
            return 1

    return 0


def clean_data(data: pd.DataFrame):
    data['Open'] = data['Open'].fillna(1)
    data.loc[(data['DayOfWeek'] == 6) & (data['Open'].isnull()), 'Open'] = 0

    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)

    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data['StoreType'] = data['StoreType'].map(mappings)
    data['Assortment'] = data['Assortment'].map(mappings)
    data['StateHoliday'] = data['StateHoliday'].map(mappings)

    data['StateHoliday'] = data['StateHoliday'].fillna(0)

    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(99999)
    data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].fillna(1)
    data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].fillna(1900)

    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                           (data.Month - data.CompetitionOpenSinceMonth)

    data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    data['Promo2SinceWeek'] = data['Promo2SinceWeek'].fillna(0)
    data['Promo2SinceYear'] = data['Promo2SinceYear'].fillna(0)
    data['PromoInterval'] = data['PromoInterval'].fillna(0)

    data['IsPromoMonth'] = data.apply(is_promo_month, axis=1)

    data['day_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

    data['month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    if 'Sales' in data.columns:
        data['SalesLog'] = np.log1p(data['Sales'])

    return data.sort_values(by=['Date'])

