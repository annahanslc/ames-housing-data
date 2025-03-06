# Ames Housing Project

!['Ames Housing](https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png)

# About This Project

Ames is a charming, urban city in Iowa, home to Iowa State University. Located approximately 30 miles north of the state capital, it is renowned for its beautiful great outdoors and low crime rate. These qualities make it an attractive destination for new residents. 

The Ames City Assesor is responsible for assessing all real property at 100% of its market value. The market value is an estimate of the price that it would sell for on the open market. 

The goal of this project is to generate a regression model that will allow the Ames City Assessor to predict the market value of any home in Ames City. 

# Directory

1. [About the Dataset](#-about-the-dataset)
2. [Data Preprocessing](#-data-preprocessing)
3. [Model Selection](#-model-selection)
4. [Model Analysis](#-model-analysis)
5. [Summary](#-summary)
6. [Next Steps](#-next-steps)


# About the Dataset

The dataset can be found at: (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

There are a total of 79 features, an Id column, and the SalePrice column. Id is an unique identifier for each transaction, and the SalePrice is the target column that the model aims to predict.

### 🔠 Features:

Based on the descriptions of the features, which can be found here: (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), the features can be categorized as follows. 

- 31 Categorical Nominal Features: cover a wide range of characteristics, everything from the shape of the property (LotShape) to zoning classification (MSZoning).

- 17 Categorical Ordinal Features: the condition of the home, on a scale, such as the quality of the kitchen from Poor to Excellent (KitchenQual).

- 19 Numeric Continuous Features: most pertain to area, such as the number of square feet in the basement (TotalBsmtSF), and the square footage of above ground living area (GrLivArea).

- 12 Numeric Discrete Features: frequently quantities, such as how many full bathrooms there are above grade (FullBath), or the number of fireplaces (Fireplaces).

### 📈 Feature Correlations:

##### Correlation to the target, SalePrice
The features with the highest correlations to the SalePrice include: 
- OverallQual @ 79.1%
- Neighborhood @ 73.9%
- GrLivArea @ 70.9%
- ExterQual @ 69.1%

The below barplot summarizes the correlations of each individual feature to the SalePrice.
![corr_feats_saleprice](https://github.com/user-attachments/assets/1dbe0224-5316-4f0d-8d2b-3fa909047420)

##### *Multicollinearity in Garage Features*
The garage features show multicollinearity, as seen in the below correlation heatmap. 
I will use models that perform feature selection to address the multicollinearity. 

![corr_garage_heatmap](https://github.com/user-attachments/assets/1e78df91-48ec-4bd8-8bec-ab4f1cbc9951)


##### *Multicollinearity in the Basement Features*
The basement quality features show multicollinearity, as seen in the below correlation heatmap. 
Again, I will use models that perform feature selection to address the multicollinearity. 

![corr_bsmt_heatmap](https://github.com/user-attachments/assets/82e531aa-bea1-4831-9579-816ff767705f)


##### *Others Multicollinearity to Notes*
- Zoning & Neighborhood
- RoofStyle and RoofMatl
- Exterior1st and Exterior2nd


# Data Preprocessing

### 🪈 Pipeline Overview:

![lasso_gs_pipeline](https://github.com/user-attachments/assets/74d4f8b2-138c-485d-b67e-3d95959881a3)

### 🚫 Treatment of Nulls:
Training dataset contains missing values in 19 features. Incoming new data are likely to contain missing values as well, so all features are equipped with an imputation method for nulls. 

### 👷 Engineered Features:
1. RatioBathBed = the ratio of the total number of bathrooms to the number of bedrooms above ground
2. HouseAge = the age of the home at the time of the sale
3. TotalBaths = total number of bathrooms on all floors, and half baths added as 0.5 bath
4. FireBedRatio = the ratio of the number of fireplaces to the number of bedrooms above ground

### 🦄 Outliers:
Features that exhibit an extreme right skew include: 'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea','OpenPorchSF'.
From these features, I removed observations with outliers that are more than 5 times the IQR in order to reduce noise, while maintaining as many observations as possible.

### 📐 Encoding and Scaling:
Categorical nominal features are OneHotEncoded, categorical ordinal features are OrdinalEncoded.
For numeric feature, I used the following scalers:
1. StandardScaler - approximately normal distributions
2. RobustScaler - skewed distirbutions and/or many outliers
3. MinMax - year, to maintain the relative difference between years

# Model Selection

### Log of Target
The target, SalePrice, has a strong right skew (left). 
To improve model performance, I took the log of the SalePrice to normalize the distribution (right).

![SalePrice_distribution](https://github.com/user-attachments/assets/abf9761e-9d05-4f64-a480-3a812d7ad7bc) ![SalePrice_log_distribution](https://github.com/user-attachments/assets/06fac3a4-994b-4d72-a7d1-a76941cca399)

### LinearRegression
- Lasso
- Polynomial

### Ensemble
- RandomForest
- XGBoost

### GridSearchCV & RandomizedSearchCV


### Best Model & Kaggle Submission

![kaggle_submission](https://github.com/user-attachments/assets/26c45416-d87f-468b-9dc4-39b20d396b42)


# Model Analysis

![strength_coef_model](https://github.com/user-attachments/assets/65ce82e7-3280-4739-bb72-216fb7bf17d4)

### Limitations of the Model

# Summary

# Next Steps
