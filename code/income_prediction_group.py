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

# Read in CSV files and put into dataframes
dataset_training = pd.read_csv('../data/tcd-ml-1920-group-income-train.csv', low_memory=False)
dataset_predict = pd.read_csv('../data/tcd-ml-1920-group-income-test.csv', low_memory=False)

##### Pre-processing code starts here #####

# define which features are numerical, categorical or constructed
numerical_features = ['Year of Record', 'Work Experience in Current Job [years]',
        'Size of City']
categorical_features = ['Housing Situation', 'Satisfation with employer', 'Gender', 'Country',
        'Hair Color']
constructed_categorical_features = ['Body Height [cm]', 'Crime Level in the City of Employement', 'Age']

# bin intervals for constructed categories
height_bins = pd.IntervalIndex.from_tuples([(0, 155), (156, 195), (196, 300)])
crime_bins = pd.IntervalIndex.from_tuples([(0, 33), (33, 133), (133, 200)])
age_bins = pd.IntervalIndex.from_tuples([(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99), (100, 109), (110, 119), (120, 129)])

# Apply bins to data
dataset_training['Body Height [cm]'] = pd.cut(dataset_training['Body Height [cm]'], height_bins, labels=['Below','Average','Above'])
dataset_training['Crime Level in the City of Employement'] = pd.cut(dataset_training['Crime Level in the City of Employement'], crime_bins, labels=['Below', 'Average', 'Above'])
dataset_training['Age'] = pd.cut(dataset_training['Age'], age_bins, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'])

dataset_predict['Body Height [cm]'] = pd.cut(dataset_predict['Body Height [cm]'], height_bins, labels=['Below','Average','Above'])
dataset_predict['Crime Level in the City of Employement'] = pd.cut(dataset_predict['Crime Level in the City of Employement'], crime_bins, labels=['Below', 'Average', 'Above'])
dataset_predict['Age'] = pd.cut(dataset_predict['Age'], age_bins, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'])

# Remove some data that contains strings instead of floats (not so elegant)
dataset_training = dataset_training[~dataset_training['Work Experience in Current Job [years]'].str.contains('#NUM!')]
dataset_training['Yearly Income in addition to Salary (e.g. Rental Income)'].str.rstrip('%')
dataset_predict['Yearly Income in addition to Salary (e.g. Rental Income)'].str.rstrip('%')

# additional_income_training = dataset_training['Yearly Income in addition to Salary (e.g. Rental Income)'].to_numpy()
# additional_income_predict = dataset_prediction['Yearly Income in addition to Salary (e.g. Rental Income)'].to_numpy()

# Data pipelines, better to substitute with individual preprocessing steps perhaps
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

##### Pre-processing code ends here #####

# system pipeline
regressor = Pipeline(steps=[
    ('Preprocessor', preprocessor),
    ('XGBoost', xgb.XGBRegressor())],
    verbose=True)

# Create features and target vectors
x_data = dataset_training.drop(['Instance', 'Yearly Income in addition to Salary (e.g. Rental Income)','Total Yearly Income [EUR]'], axis=1)
y_data = dataset_training['Total Yearly Income [EUR]']

x_pred = dataset_predict.drop(['Instance', 'Yearly Income in addition to Salary (e.g. Rental Income)','Total Yearly Income [EUR]'], axis=1)

# validation/train split
x_train, x_test, y_train, y_real = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

model = regressor.fit(x_train, y_train)

y_test = model.predict(x_test)
y_pred = model.predict(x_pred)

# write to csv, can improve so it writes out submittable file
predictions = pd.DataFrame(y_pred)
predictions.to_csv('../pred.csv')

# Print MAE
print('MAE: ', mean_absolute_error(y_test, y_real))
