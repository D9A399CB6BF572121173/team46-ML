import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

dataset_training = pd.read_csv('../data/tcd-ml-1920-group-income-train.csv', low_memory=False)
dataset_predict = pd.read_csv('../data/tcd-ml-1920-group-income-test.csv', low_memory=False)

numerical_features = ['Year of Record', 'Work Experience in Current Job [years]', 'Age',
        'Size of City']
categorical_features = ['Housing Situation', 'Satisfation with employer', 'Gender', 'Country',
        'Hair Color']
constructed_categorical_features = ['Body Height [cm]', 'Crime Level in the City of Employement']

for feature in constructed_categorical_features:
    dataset_training[feature] = pd.cut(dataset_training[feature], 3, labels=['Below','Average','Above'])

dataset_training = dataset_training[~dataset_training['Work Experience in Current Job [years]'].str.contains('#NUM!')]

numerical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median', verbose=1)),
    ('Scaler', StandardScaler())],
    verbose=True)

categorical_transformer = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent',verbose=1)),
    ('Encode', OneHotEncoder(handle_unknown='ignore'))],
    verbose=True)

preprocessor = ColumnTransformer(
        transformers=[
            ('Numerical Data', numerical_transformer, numerical_features),
            ('Category Data', categorical_transformer, categorical_features),
            ('Binned Data', categorical_transformer, constructed_categorical_features),],
        verbose=True)

regressor = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('XGBoost', xgb.XGBRegressor())],
    verbose=True)

x_data = dataset_training.drop(['Instance', 'Yearly Income in addition to Salary (e.g. Rental Income)',
    'Total Yearly Income [EUR]'], axis=1)
y_data = dataset_training['Total Yearly Income [EUR]']

x_train, x_test, y_train, y_real = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

model = regressor.fit(x_train, y_train)
y_pred = model.predict(x_test)

predictions.pd.DataFrame(data=regressor.predict(y_pred), columns='Total Yearly Income [EUR]')
predictions.insert(0, 'Instance', range(1, len(df)))
predictions.to_csv('../pred.csv')

print('MAE: ', mean_absolute_error(y_real, y_pred))
