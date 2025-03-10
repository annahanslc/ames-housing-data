########################################################################
# CUSTOM HELPER FUNCTIONS
########################################################################

# Function for evaluating different regression models

from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error,
r2_score, mean_absolute_percentage_error)

import numpy as np

def adj_r2(r2, x):
  n = x.shape[0]
  p = x.shape[1]
  return 1 - (((n-1) / (n - p- 1)) * (1 - r2))

def evaluate_regression(model, X, y, name='model'):
  y_pred = model.predict(X)
  MAPE = mean_absolute_percentage_error(y, y_pred)
  r2 = r2_score(y, y_pred)
  RMSE = root_mean_squared_error(np.exp(y), np.exp(y_pred))
  MSE = mean_squared_error(y, y_pred)
  MAE = mean_absolute_error(y, y_pred)
  a_r2 = adj_r2(r2, X)
  metrics = ['MAE','MSE','RMSE','MAPE','R2','adj_r2']
  results = pd.DataFrame(columns=metrics, index=[name])
  results['MAE'] = [MAE]
  results['MSE'] = [MSE]
  results['RMSE'] = RMSE
  results['MAPE'] = MAPE
  results['R2'] = r2
  results['adj_r2'] = a_r2

  return results

########################################################################

# Custom transformer for dropping outliers

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
  def __init__ (self, columns, iqr_multiplier=1.5):
    """Calculate outliers based on IQR times a multiplier.

    column: list of columns to check for outliers
    iqr_multiplier: set the outlier range, defaults to 1.5"""

    self.columns = columns
    self.iqr_multiplier = iqr_multiplier

  def fit(self, X, y=None):
    #Calculate the IQR and thresholds for outlier detection.
    self.thresholds_ = {}

    for column in self.columns:
      if column not in X.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

      Q1 = X[column].quantile(0.25)
      Q3 = X[column].quantile(0.75)
      IQR = Q3-Q1

      lower_threshold = Q1 - (IQR * self.iqr_multiplier)
      upper_threshold = Q3 + (IQR * self.iqr_multiplier)

      self.thresholds_[column] = (lower_threshold, upper_threshold)

    return self

  def transform(self, X, y=None):
    #Remove outliers based on the calculated thresholds.
    X = X.copy()

    mask = pd.Series(True, index=X.index)

    for column, (lower_threshold, upper_threshold) in self.thresholds_.items():
      mask &= X[column].isna() | (X[column] >= lower_threshold) & (X[column] <= upper_threshold)

    return X[mask].reset_index(drop=True)



########################################################################

# Function for checking for outliers

def check_outliers(data, column, iqr_multiplier=1.5):
  # calculate 25% and 75% quantile
  qt_25 = data[column].quantile(0.25)
  qt_75 = data[column].quantile(0.75)

  # calculate the interquartile range
  iqr = qt_75 - qt_25

  # calculate the lower and upper thresholds
  lower = qt_25 - iqr*iqr_multiplier
  upper = qt_75 + iqr*iqr_multiplier

  data_wo_outliers = data[(data[column] >= lower) & (data[column] <= upper)]
  data_outliers = data[(data[column] < lower) | (data[column] > upper)]

  num_outliers = data_outliers[column].count()
  num_data = data[column].count()

  percent_outlier = '{:.2%}'.format(num_outliers / num_data)


  print(f'The original dataframe contains {num_data} observations.')
  print(f'Using IQR * {iqr_multiplier}, {num_outliers} outliers were detected.')
  print(f'If removed, {percent_outlier} of the data will be dropped.')

########################################################################
