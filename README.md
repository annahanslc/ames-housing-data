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
The below barplot shows the correlations of each individual feature to the SalePrice, sorted from highest to lowest, based on the standard Pearson correlation, r.
![corr_feats_saleprice](https://github.com/user-attachments/assets/1dbe0224-5316-4f0d-8d2b-3fa909047420)

The features with the 20 highest correlations to the SalePrice are: 

| Feature       |   Corr to SalePrice |
|:--------------|--------------------:|
| OverallQual   |           0.790982  |
| Neighborhood  |           0.73863   |
| GrLivArea     |           0.708624  |
| ExterQual     |           0.690933  |
| BsmtQual      |           0.681905  |
| KitchenQual   |           0.675721  |
| GarageCars    |           0.640409  |
| GarageArea    |           0.623431  |
| TotalBsmtSF   |           0.613581  |
| 1stFlrSF      |           0.605852  |
| FullBath      |           0.560664  |
| GarageFinish  |           0.553059  |
| FireplaceQu   |           0.542181  |
| TotRmsAbvGrd  |           0.533723  |
| YearBuilt     |           0.522897  |
| YearRemodAdd  |           0.507101  |
| Foundation    |           0.506328  |
| GarageType    |           0.499204  |
| MasVnrArea    |           0.472614  |
| Fireplaces    |           0.466929  |


##### *Observations*
1. 🥰 **Quality** indicators, including OverallQual, ExterQual, BsmtQual, KitchenQual, and FireplaceQual have strong correlations to SalePrice. These indicators are on a range from Poor to Excellent, which required a certain amount of subjective judgement from an observer. Although it is a subjective decision, or perhaps, it is precisely because it is someone's opinion, they are highly correlated to the SalePrice. This tells us that:
    1. A buyer of the home will likely exhibit a similar judgement call on the quality on the home, and be possibly swayed, and/or be emotionally affected, by things that are difficult to quantity using numbers or statistics.
    2. Quality features should not be overlooked due to their subjective nature, but instead, can be further expanded to get an even better understanding of the home's selling value.
    3. Since there are currently no features in the dataset to capture the quality of bathrooms, I would recommend adding BathsOverallQual to the data gathering process, and then incorporate it into this model.

2. 🏡 **Neighborhood** is strongly correlated to the SalePrice. This indicates that homes in certain areas of the city tend to sell at higher or lower prices than those in other parts of the city. This also means that some of the best predictors of a home's selling price are the prices at which other homes nearby sold for. This reinforces the idea that comps (comparable sales nearby) are effective in predicting a home's selling price. 

3. 📏 Not surprisingly, **area** (measured in square feet), correlates highly with the SalePrice. The bigger the home, the more expensive it tends to be. There are several features that lend to gauging the size of the home. In my feature engineering, I hope to find way new ways to capture the level of "luxury" of the home, rather than just the size. 

4. 🚗 **GarageCars** and **GarageArea** have similar levels of correlation with SalePrice. This makes sense, as the the number of cars a garage can hold increases with the area of the garage. I will calculate the correlation between these two variables, as well as other garage-related features, to check for multicollinearity, as this will affect my feature selection and/or model selection.


##### *Multicollinearity in Garage Features*

The below correlation heatmap shows the correlation between the various garage-related features. 

![corr_garage_heatmap](https://github.com/user-attachments/assets/1e78df91-48ec-4bd8-8bec-ab4f1cbc9951)

Observation: as expected, the garage features do show multicollinearity, as indicated by the entirely warm-colored cells in the heatmap. However, the correlation of 1.0 between GarageYrBlt and GarageType is unexpected. To investigate, I will map the correlation between GarageYrBlt and OneHotEncoded values of GarageType. 

![corr_garagetype_garageyrblt](https://github.com/user-attachments/assets/eea672e6-a25d-45e4-8357-b22d8936d805)

Based on the above heatmap, we can see that the perfect correlation between GarageYrBlt and GarageType is due to the observations that have nulls in both features. This makes sense because if the home does not have a garage, then both the GarageType and the GarageYrBlt will be null. GarageYrBlt is a numeric value, but because it represents year, and not a quantity or quality, most numeric imputing strategies would not be meaningful. For most homes, the garage is built the same year as the home, so the best approximation for GarageYrBlt would be YearBuilt. However, this means that if a home does not have a garage, this is an important aspect of the home will be lost. Fortunately, there are several other features that will still capture this fact: GarageArea & GarageCars would be 0 and GarageType, GarageQual & GarageCond would all be NA. 

Based on the above observations, I will:
1. Impute GarageYrBuilt nulls with YearBuilt
2. Use models that perform feature selection to address multicollinearity. 

##### *Multicollinearity in the Basement Features*

The below correlation heatmap shows the correlation between the nominal basement features.

![corr_bsmt_heatmap](https://github.com/user-attachments/assets/82e531aa-bea1-4831-9579-816ff767705f)

As expected, there are notable correlations between these features. Based on my investigation on the garage features, I am confident that most of these correlations are driven by the fact that the homes that do not have a basement will have the same values, NA, across all features. Due to this unavoidable characteristic, basement nominal features show multicollinearity. Again, I will use models that perform feature selection to address the multicollinearity. 

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
1. RatioBathBed = the ratio of the total number of bathrooms to the number of bedrooms above ground, to capture the level of luxury of the home
2. HouseAge = the age of the home at the time of the sale, as newer homes tend to cost more
3. TotalBaths = total number of bathrooms on all floors, and half baths added as 0.5 bath, signals the size of the home
4. FireBedRatio = the ratio of the number of fireplaces to the number of bedrooms above ground, again, another indicator of luxury

### 🦄 Outliers:
Features that exhibit an extreme right skew include: 
'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea','OpenPorchSF'.

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
- Linear Regression - best log RMSE @ 0.34314
- Lasso - best log val RMSE @ 0.13223

### Ensemble with GridSearchCV & RandomizedSearchCV
- RandomForest - best log val RMSE @ 0.21229
- XGBoost - best log val RMSE @ 0.16118


### Best Model & Kaggle Submission

Lasso optimized using GridSearchCV returned the best results:
![image](https://github.com/user-attachments/assets/a2eae525-8441-479e-bc20-04a033e1f9ee)

![kaggle_submission](https://github.com/user-attachments/assets/26c45416-d87f-468b-9dc4-39b20d396b42)


# Model Analysis

### The top 10 coefficients in the final model are as follows:

1. RoofMatl_ClyTile @ 0.296
2. GrLivArea @ 0.122
3. Neighborhood_StoneBr @ 0.089
4. Neighborhood_Crawfor @ 0.085
5. Neighborhood_NridgHt @ 0.082
6. Exterior1st_BrkFace @ 0.078
7. MSSubClass_160 @ 0.074
8. OverallQual @ 0.061
9. GarageCars @ 0.055
10. Neighborhood_Somerst @ 0.048

![strength_coef_model](https://github.com/user-attachments/assets/65ce82e7-3280-4739-bb72-216fb7bf17d4)




# Summary

# Next Steps
