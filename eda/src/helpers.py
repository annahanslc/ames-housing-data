########################################################################
# CUSTOM HELPER FUNCTIONS
########################################################################

# Function for evaluating different regression models

from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error,
r2_score, mean_absolute_percentage_error)

def adj_r2(r2, x):
  n = x.shape[0]
  p = x.shape[1]
  return 1 - (((n-1) / (n - p- 1)) * (1 - r2))

def evaluate_regression(model, X, y, name='model'):
  y_pred = model.predict(X)
  MAPE = mean_absolute_percentage_error(y, y_pred)
  r2 = r2_score(y, y_pred)
  RMSE = root_mean_squared_error(y, y_pred)
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

from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
  def __init__ (self, column, iqr_multiplier=1.5):
    """Calculate outliers based on IQR times a multiplier.

    column: name of column to check for outliers
    iqr_multiplier: set the outlier range, defaults to 1.5"""

    self.column = column
    self.iqr_multiplier = iqr_multiplier

  def fit(self, X, y=None):
    #Calculate the IQR and thresholds for outlier detection.
    X = X.copy()
    Q1 = X[self.column].quantile(0.25)
    Q3 = X[self.column].quantile(0.75)
    IQR = Q3-Q1

    self.lower_threshold_ = Q1 - (IQR * self.iqr_multiplier)
    self.upper_threshold_ = Q3 + (IQR * self.iqr_multiplier)
    return self

  def transform(self, X, y=None):
    #Remove outliers based on the calculated thresholds.
    X = X.copy()

    #Return error if column does not exist in dataframe
    if self.column not in X:
      raise ValueError(f"Column '{self.column}' not found in Dataframe")

    #Filter out observations with outliers, keep other
    mask = (X[self.column] >= self.lower_threshold_) & (X[self.column] <= self.upper_threshold_)
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
