# Ames Housing Project

!['Ames Housing](https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png)

# About This Project

Ames is a charming, urban city in Iowa, home to Iowa State University. Located approximately 30 miles north of the state capital, it is renowned for its beautiful great outdoors and low crime rate. These qualities make it an attractive destination for new residents. 

The Ames City Assesor is responsible for assessing all real property at 100% of its market value. The market value is an estimate of the price that it would sell for on the open market. 

The goal of this project is to generate a regression model that will allow the Ames City Assessor to predict the market value of any home in Ames City. 

Throughout the project, I will gauge the accuracy of my model by calculating the log RMSE of my validation dataset. Ultimately, the success of the model will be determined by Kaggle in the form of log RMSE of my predictions. 

# Process Overview

The below outlines the steps I will take in this project. 

1. Conduct Exploratory Data Analysis (EDA) to understand the features individually. During my EDA, I will also:
    1. Address nulls and determine imputation stratgies
    2. Check for outliers and determine if any need to be removed
    3. Check for bad data
    4. Determine if the feature needs to be encoded or scaled
    5. Analyze features for correlations and multicollinearity
2. Engineer new features to capture new perspectives on the data to provide additional information to the model.
3. Build a data preprocessing pipeline to easily fit and transform data based on my discoveries during EDA. My pipeline will include the following components:
    1. Outlier Remover Custom Transformer
    2. Feature Engineering Custom Transformer
    3. Imputers using various strategies
    4. Encoders for OneHotEncoding and OrdinalEncoding
    5. Scaler for scaling numeric features.
4. Fit various models and calculate the log RMSE to determine the best model for this project
5. Use GridSearchCV and RandomizedSearchCV to fine tune the model's hyperparameters.
6. Once the final model is selected, analyze the model to understand what features are the strongest drivers.
7. Make plans for next steps and future improvements to the model.
   

# Directory

1. [About the Dataset](#-about-the-dataset)
3. [Data Preprocessing](#-data-preprocessing)
4. [Model Selection](#-model-selection)
5. [Model Analysis](#-model-analysis)
6. [Summary](#-summary)
7. [Next Steps](#-next-steps)


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

### Correlation to the target, SalePrice
To get a general idea of the correlations within the dataset prior to preprocessing, I will calculate the correlation of all features to the target, SalePrice. Since most features are categorical, I will use the *dython* library's nominal associations function to help determine the Pearson correlation coefficients (r). 

However, I must keep in mind that dython associations treat ordinal features as nominal, meaning that is does not understand the order in the values. I will not have the complete picture until I encode these features using the Ordinal Encoder. 

The below barplot shows the correlations sorted from highest to lowest.

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


### *Observations of Features that Correlate Highly with SalePrice*
1. 🥰 **Quality** indicators, including OverallQual, ExterQual, BsmtQual, KitchenQual, and FireplaceQual have strong correlations to SalePrice. These indicators are on a range either from 1-10 or from Poor to Excellent, which required a certain amount of subjective judgement from an observer. Although it is a subjective decision, or perhaps, it is precisely because it is someone's opinion, they are highly correlated to the SalePrice. This indicates that:
    1. A buyer of the home will likely exhibit a similar judgement call on the quality on the home, and be possibly swayed, and/or be emotionally affected, by things that are difficult to quantity using numbers or statistics.
    2. Quality features should not be overlooked due to their subjective nature, but instead, can be further expanded to get an even better understanding of the home's selling value.
    3. Since there are currently no features in the dataset to capture the quality of bathrooms, I would recommend adding BathsOverallQual to the data gathering process, and then incorporate it into this model.
    4. Again, because dython associations does not account for the order in ordinal features, these correlations may not paint the full picture, and could even be misleading. 
  
    ![sns_overallqual](https://github.com/user-attachments/assets/77e32b17-63b3-4f44-b900-e1d31de64dfe)

    5. The above *seaborn stripplot* visualizes the correlation between OverallQual and the SalePrice. It is clear that a higher overall quality corresponds to a higher sales price. However, the distribution is more sparse and varied on the higher end of the scale. This means that there are fewer samples with an OverallQual of 9 or 10, and that the correlation to sale price decreases when the OverallQual is extremely high. 

2. 🏡 **Neighborhood** SalePrice and Neighborhood has a Pearson's coefficient of 0.73863, which indicates a strong, meaningful, positive correlation. This suggests that similar homes in certain areas of the city tend to sell at higher or lower prices than those in other parts of the city. To help me better understand the impact of neighborhoods on SalePrice, I will use neighborhood to subset the feature that have the highest correlations besides Neighborhood, OverallQual:

    1. *SalePrice vs OverallQual by Neighborhood* Since OverallQual has the highest correlation with SalePrice, I will plot it against SalePrice, and separate it into subsets by neighborhood. The below **seaborn FacetGrid** plot shows a linear regression line for **each neighborhood**. The shaded area surrounding each line is the confidence interval, which how confident the model is about the range of values. The plot implies that having a high overall quality has more positive correlation with SalePrice in some neighborhoods than in others. While an Excellent OverallQual correlated to a much higher SalePrice in NoRidge (Northridge), it didn't have much effect in IDOTRR (Iowa Department of Transportation). This highlights the important role that neighborhood plays in helping the model to accurately predict the SalePrice.
  
       ![sns_facet_neighborhood_grlivarea](https://github.com/user-attachments/assets/bfc5aed4-c76e-401c-b895-8ccaa16bb919)

4. 📏 Not surprisingly, **area** (measured in square feet), correlates highly with the SalePrice. The bigger the home, the more expensive it tends to be. There are several features that lend to gauging the size of the home. In my feature engineering, I will try to find ways to capture the level of "luxury" of the home, not just the size. But first, I will take a closer look at the area feature that has the highest correlation with sale price, Ground Living Area. 

    1. **GrLivArea** has the 3rd highest correlation with SalePrice, so I used the below **seaborn regplot** to visualize the relationship between the two. As expected, the regplot shows a positive correlation between SalePrice and GrLivArea. The line represents the best-fit linear regression model, and the shaded area is the confidence interval, which represents the level of uncertainty of the model. The confidence interval here is fairly narrow, which means that model is fairly confident. How tightly the datapoints are clustered around the regression line speaks to the strength of the relationship. The observations surround the line, however, they are not tightly clustered, so GrLivArea is in no way a perfect predictor of SalePrice. This is especially evident with the outliers that have a large GrLivArea, but do not fetch a high SalePrice.

    ![sns_saleprice_grlivarea](https://github.com/user-attachments/assets/3b3f1c27-8f2f-4538-9a60-1e8c42eb9dbf)


5. 🚗 **GarageCars** and **GarageArea** have similar levels of correlation with SalePrice. This makes sense, as the the number of cars a garage can hold increases with the area of the garage. I will calculate the correlation between these two variables, as well as other garage-related features, to check for multicollinearity, as this will affect my feature selection and/or model selection.


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


### *Observations of Other Features*

1. **Building Type** I expect that the type of building would be correlated to the SalePrice as well. The Pearson R correlation coefficient for SalePrice and BldgType is 0.19, which is not as high as some of the other features, but still notable. To visualize what kind of impact it has on SalePrice, I will use *seaborn's lmplot* to plot SalePrice against GrLivArea using BldgType as the hue.

   ![sns_price_area_bldgtype](https://github.com/user-attachments/assets/8e50e986-23da-45a4-8848-1c3de3c0afb2)

The above plot leads to a few interesting observations:

- The slope is much steeper for single family homes (1Fam) than for two-family conversion homes (2FmCon). Two-family conversion homes were originally built as one-family dwelling, but later converted into 2 dwellings. The steeper slope means that there is a stronger relationship between SalePrice and GrLivArea for 1Fam than for 2FamCon. This indicates that even if the homes are the same size, a 2FmCon correlates to a lower SalePrice than a 1Fam. This does not necessarily mean that converting a single family home into a 2 family conversion will decrease the SalePrice, because correlation does not mean causation, and there may be confounding factors, but this would be something interesting to look into for a future project.
- The size and the price of a two-family conversion are limited. Whereas single-family homes have a wide range of values, with outliers in all directions, two-family conversions are limited in size and price. Based on my research, two-family conversions are typically sold as 1 single building. This means that the types of single family homes that are usually converted into two-family dwellings are not too big. This makes sense, because conversions are usually conducted in high-density areas to meet housing demand. It seems unlikely that a high-end luxury home would ever be converted.
- The regression lines for 2famCon and Duplex are almost the same. This signals that the correlation between area and price is similar for these two building types. This is unsurprising, because 2famCon is basically a Duplex, just converted after the fact.
- Townhouse End Units (TownhsE), interestingly, have the steepest slope. This means that they exhibit the highest price per square feet. However, their area is limited, so we don't see any high square footages that can be observed in single family homes.

2. **Garage Finish** I wonder if the different levels of Garage Finishes have different correlations with Sale Price. I will use the *seaborn displot* to visualize the 3 different types of Garage Finishes: Finished (Fin), RFn (Rough Finish), Unf (Unfinished), as well as "None", for no garage.

   ![sns_garagefinishes](https://github.com/user-attachments/assets/6e263c3b-00ad-4b11-9816-a06674c70779)


- The plot shows notably different distributions for the 4 garage types. The homes that don't have a garage correlate to sale prices that are much lower than those with a garage. The lines on the plot are a smooth representation of the data distribution. The peaks of each kde plot line is the value with the highest frequency, or the mode. To better compare the modes for each finish type, I will use a groupby to filter the dataset by the type of GarageFinish, then aggregate the SalePrice column, in order use a lambda function to calculate the mode for each type of garage finish. The below table shows the calculated modes:
       

   | GarageFinish   |   SalePrice   |
   |:---------------|--------------:|
   | No Garage      |       $84,500 |
   | Unf            |      $135,000 |
   | RFn            |      $190,000 |
   | Fin            |      $185,000 |
   

- In line with the displot, the lowest mode is No Garage, and the next one is Unfinished Garage. Surprisingly, the next highest is Finished Garage, not Rough Finish. I did not expect the Rough Finish to have a higher mode than Finished. However, looking back to the displot, finished garage homes have a fatter tail towards the higher end of Sale Price. This indicates that although the mode may be lower, there is a greater chance for a home with a finished garage to correlate to a higher sale price than a home with a rough finished garage.

3. **Kitchen Quality** To visualize the distribution of sale price at various levels of kitchen quality, I will use the *seaborn violin plot* together with the *swarm plot*. The violin plot is able to show the distribution density and shape using KDE, while the swarm plot shows the real data points, which in turn, allows us to compare the sample sizes.

    ![sns_kitchenqual](https://github.com/user-attachments/assets/0cb36de7-3e08-42a3-ae19-9e3b41652e8f)

- From the above plot, it is notable that the sample size for Excellent and Fair are much smaller than those for Good and Typical/Average.
- Overall, there is definitely a positive correlation between a better quality kitchen and the sale price of the home.
- It is interesting to see that an Excellent quality kitchen correlates to a very wide range of sales prices.
- Like I mentioned earlier, the quality indicators offer a unique perspective on the home, because these can capture aspects of the home that may not be quantifiable, but can still influence the purchasing behavior or a home buyer, thereby, affecting the sale price.

4. **Lot Shape** I am curious what kind of correlation exists between sale price and lot shape. My initial impression is that an irregular lot shape will correspond to a lower sale price. I will check my hypothesis by using the *seaborn boxenplot* to visualize the distribution of sale price for each level of lot irregularity. The 4 levels are: Reg = Regular, IR1 = Slightly irregular, IR2 = Moderately Irregular, IR3 = Irregular.

   ![sns_lotshape](https://github.com/user-attachments/assets/2c626b2e-e192-4822-82d0-9143fb368dd8)

- Based on the above plot, contrary to my prediction, the various levels of irregular lots correspond to higher sales prices than the regular lot.
- This may be because larger lots are more likely to be irregular, and larger lots correspond to higher prices. However, this is only a theory and additional analysis would be necessary in order to draw a deeper understanding of this relationship.

5. **Year Built** I would expect that in general, new homes will correspond to higher sales prices. I will check my assumption by plotting YearBuilt against SalePrice using the *seaborn lineplot*.

    ![sns_yearbuilt](https://github.com/user-attachments/assets/43ddddc9-3dfe-44ca-bc3f-50f6280b3f2f)

- As anticipated, there is a positive correlation between year built and the sale price. However, the correlation does not appear to be linear. Rather, the upward trend only starts around 1950, and homes built prior to 1950 have large fluctuations in pricing. Possible explanations could be historical or renovated buildings have a higher value. 
- Between 1950 and 2010, the correlation stabilizes into a consistent upward trend.
- There is spike in home prices in homes built after 2010. This could be because of the cost of modern amenities, and demand for newer properties.
- I will next use a residual plot to check if the correlation models well using a linear regression. Again, I suspect that the relationship is non-linear.

    ![sns_residual](https://github.com/user-attachments/assets/55c5a82b-6f00-4c7a-b99e-102a2cc26ecc)

- In the above *seaborn residplot*, the red line is the LOESS smoother, a non-parametric regression method, which means that it dynamically learns patterns from data instead of using a set function. If LOESS line is flat at 0, it would mean that residuals are randomly scattered and that there is no systematic bias. However, a curve in the LOESS line, as seen in the above plot, indicates that there is a non-linear relationship. 
- For homes built before 1960, the model increasingly overestimates home prices, this suggests that the model is unable to account for additional factors that affect sale price, possibly costs such as depreciation or major repairs that impact older homes. Around 1960-1980, the LOESS line is the flattest, which means that the model is the most stable, although there is still bias, since it is consistently overestimates the sale price. After 1980s, the model starts to increasingly underestimate sale price. This could be driven by the fact thatdemand of newer home is much higher, and/or the cost of modern amenities in newer homes drive the sale price higher.


# Data Preprocessing

### 🪈 Pipeline Overview:

![preprocessing_pipline_final](https://github.com/user-attachments/assets/935a108f-339c-4cf3-882f-16fc152309b3)


### 🚫 Treatment of Nulls:
The training dataset contains missing values in 19 features. Incoming new data is highly likely to also contain missing values in these features, but also in features that do not have nulls in the training dataset. Therefore, I have equipped all features with an imputation method for nulls. 

##### Decisions to Note:
Nulls in GarageYrBlt most likely means that there is no garage. I have several options to impute these nulls. I can:

1. Impute with 0, to signify no garage. However, since the valid values are years, which are in the thousands, 0 is not a valid year and could distort the relationship in between the valid years. The model could mistakenly interpret 0 as a very old garage.
2. Impute with a constant, such as 1900. But again, this could mislead the model to think of it as an old garage rather than no garage. Any other year I choose as the constant could sway the model, misdirecting the model from understanding the correlation between the valid years. 
3. Impute with the year the same year that the house with built in. This assumes that most garages are built in the same year that the house was built. According to Pearson coefficient, the correlation between the two is 0.826, which is very strong. The downside is that we lose the perspective of which homes don't have a garage and which ones do.  However, there are 6 other features that still capture this crucial detail: GarageType, GarageFinish, GarageCars, GarageArea, GarageQual and GarageCond. 

None of the imputation methods are ideal for this situation, but based on the above pros and cons, I will impute with the same year that the house was built in, with the goal of trying to preserve the unique information that this feature brings (age of the garage). 

### 👷 Engineered Features:
To capture new perspectives on the home, I created 11 engineered features and added them to my model, one by one. With each new feature, I examined the uplift in log RMSE. If the log RMSE improved, I kept the new feature, if it did not, I dropped the feature. Through this process, I have decided to keep the following 4 new engineered features:

1. **RatioBathBed** = the ratio of the total number of bathrooms to the number of bedrooms above ground, to capture the level of luxury of the home. The idea is that whereas having more bedrooms and bathrooms both signal a bigger home, having a higher ratio of bathrooms to bedrooms, is a sign of affluence. According to realtor.com, the cost to add a bathroom to a new home is $63,986, while the cost to add a bedroom is $62,500. Keeping in mind that bedrooms are usually at least twice the size of a bathroom, that means the cost per SF is more than double. [Go to references](#reference-1)
2. **HouseAge** = the age of the home at the time of the sale. Since the YrSold spans from 2009 to 2010, the YearBuilt is insufficient for expressing the age of the home at the time it was sold. By taking the YrSold and subtracting the YearBuilt, we can calculate the exact age of the home at the time of the sale. 
3. **TotalBaths** = the total number of bathrooms on all floors, with half baths added as 0.5 bath. There are 4 features related to the number of bathrooms: FullBath, HalfBath, BsmtFullBath, and BsmtHalfBath, however, none of them represent the total number of bathrooms. Again, the number of bathrooms signal both the size of the home as well as the level of luxury. 
4. **FireBedRatio** = the ratio of the number of fireplaces to the number of bedrooms above ground. According to Angi.com (formerly Angie's List), a website for finding local home service providers, real estate agents say homes with fireplaces often get offers above the asking price, and are especially popular with homebuyers in colder regions. The Iowa State University's Department of Statistics describes the Iowa's winters as being "Not as cold as it gets in Minnesota!" but still with a winter "average temperatures range in 10-30 F". The ratio of number of fireplaces to number of bedrooms offers another insight into the value of a home. [Go to references](#reference-2) 

### 🦄 Outliers:
Outliers can have a significant impact on data anlysis, especially on linear regression models. Because these observations are far away from the expected results, they can have a disproportionate influence on the model, increasing both model variance and bias. Outliers can also detract from selecting the best model, since the model will appear to be less accurate overall when trying to predict outliers. 

From my EDA, I found that the following features exhibit an extreme right skew: 
'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea','OpenPorchSF'.

My goal is the reduce the negative impact of outliers in my model while preserving as many observations as possible. From the above mentioned features, I removed observations with outliers that are more than **5 times the IQR** in order to reduce noise. This is a fairly generous threshold for outliers, because again, I hope to keep as many observations in my train and val datasets as I can. The remaining outliers I will address through scaling and model selection.

Another way to tame these outliers would be to use a **log transformation**. The log transformation would help stabilize variance and normalize the distribution. I conducted log transformations on these features as well, but there was no uplift from my final model which uses the RobustScaler to address the skewed distribution. 

### 📐 Encoding and Scaling:
I used the **OneHotEncoder** to encode the categorical nominal features, and the **OrdinalEncoder** to encode the categorical ordinal ones. The OneHotEncoder transforms categorical variables into numbers by creating a binary for each category in each feature, where '1' means the category is applicable, and '0' means it is not. The OrdinalEncoder works similarly, but for when there is a meaningful order to the categories, and we want to encode them into numbers that correspond to that order. I used a list of dictionaries, where each dictionary contains the keys 'col' for column name and 'mapping' for the ordering instructions. 

For example, most quality features are ranked on a scale from Poor to Excellent, or NA if the feature is not applicable. The full order, including all possible values would be dict_na_ex_6 = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

For the numeric features, I used the **StandardScaler**. The goal of scaling is to ensure that all features contribute equally to the model. Otherwise, features with larger ranges or numbers, can overpower the model and distract it from considering all relevant data equally. This is especially important in this dataset because there are several different units of measure that have different ranges. For example, SalePrice is in dollars, and ranges from a minimum of $34,900 to a maximum of $755,000, while YrSold is in year and ranges only from 2009 to 2010.
 
# Model Selection

### Log of Target
The target, SalePrice, has a strong right skew (left). To improve model performance, I took the log of the SalePrice to normalize the distribution (right). Transforming the target helps to stabilize the variance and reduce heteroscedasticity. Heteroscedasticity is the observation that errors are inconsistent over observations. Taking the log of the SalePrice also helps to linearize any exponential relationships between variables, making it more suitable for linear models. 

![SalePrice_distribution](https://github.com/user-attachments/assets/abf9761e-9d05-4f64-a480-3a812d7ad7bc) ![SalePrice_log_distribution](https://github.com/user-attachments/assets/06fac3a4-994b-4d72-a7d1-a76941cca399)

I explored several regression models in search of the best model for this dataset. 

1. **Linear Regression** Using the LinearRegression function provided by scikit-learn's linear_model library, I generated a linear regression model to predict the SalePrice. The linear regression model describes the relationship between a dependent variable, y, and one or more independent variables, X. In this dataset, y is the SalePrice and the features are the independent variables, X's. The resulting model yielded the following performance:

   | Metric         |     Result    |
   |:---------------|--------------:|
   | train RMSE     |       $21,016 |
   | val RMSE       |       $30,771 |
   | variance       |        $9,755 |
   | log val RMSE   |        0.1975 |

- The above metrics indicate that the model performed much better on train than on val. The variance of $9,755 demonstrates that the model is overfitted. The log val RMSE is not bad, but not where I want to be. This linear regression model is my simpliest model, and will act as my baseline. I will next try more complex models and compare their performances against my baseline.

2. **Linear Regression with Lasso and GridSearchCV** scikit-learn's Lasso function, also from the linear_model library, helps to prevent overfitting by conducting regularization. Regularization is the technique used to prevert overfitting by penalizing large coefficient values, so it discourages the model from fitting the training data too closely. Lasso uses L1 Regularization, which utilizes the absolute values of the coefficients to calculate the penalty. Lasso can be tuned by adjusting its regularization strength, alpha. I will use GridSearchCV to run the model using different alphas, and find the parameters that generates the best model.

GridSearchCV is a technique from scikit-learn that systematically searches for the best combination of hyperparameters by trying and evaluating all possible combinations. It also use cross-validation to determine the best performance. Cross-validation is when a dataset is split into train and test data in multiple ways to avoid modeling or testing off of a single "lucky draw". This can improve the relability of the performance metric of the model. 

The best model resulting from the Lasso regression using GridSearchCV yielded the following metrics:

   | Metric         |     Result    |
   |:---------------|--------------:|
   | train RMSE     |       $26,583 |
   | val RMSE       |       $27,675 |
   | variance       |        $1,092 |
   | log val RMSE   |        0.1322 |

- The Lasso model performed much better in terms of the val RMSE, variance, and log RMSE. It performed more poorly on train RMSE, but that is not much of a concern, since our business goal is to find the best model to predict future sale price, not the train dataset. I am especially happy with the lower variance. This lends confidence to the model in its ability to accurately predict real data in the future. The Lasso has successfully surpassed the baseline model and is a contender for the final model. 

3. **RandomForestRegressor with GridSearchCV** RandomForest is an ensemble model that utilizes multiple decision trees to improve overall accuracy and prevent overfitting. It does this by building many decision trees, and then averaging their predictions to produce a final output. This can help capture complex patterns and relationships in between features. The RandomForest model is robust against outliers, and is well suited for dataset with a large number of features.

Instead of Regularization, RandomForestRegressor uses hyperparameters such as max_depth, min_samples_split, min_samples_leaf and max_features to help control tree complexity, which helps to avoid overfitting. 

The best model that resulted from the GridSearchCV on the RandomForestRegressor yielded the following results:

   | Metric         |     Result    |
   |:---------------|--------------:|
   | train RMSE     |       $11,433 |
   | val RMSE       |       $29,238 |
   | variance       |       $17,805 |
   | log val RMSE   |        0.1434 |

- The variance on the RandomForestRegressor model is the highest so far, this signals that the model is very overfitted. The val RMSE is smaller than the linear regression model, but higher than Lasso. The log val RMSE is also higher. Comparatively, the Lasso model performed better, and is still the best model thus far. 

4. **XGBoost with GridSearchCV** XGBoost is a newer model that utilizes gardient boosting, which is when an ensemble of decision trees are built sequentially. Each tree's goal is to correct the errors of the previous ones by minimizing a loss function, such as the mean squared error. It is also excellent at handling complex data patterns, AND it has built-in regularization by using L1 and L2 penalties on leaf weights.

There are a number of hyperparameters that can be tuned in an XGBoost model. To help curb overfitting, I will adjust the range of parameters provided to the GridSearchCV to reduce the complexity of the tree. Specifically, I will:

1. Limit the number of trees using n_estimators
2. Limit the depth of each tree, which helps to control complexity
3. Increase the min_child_weight, since larger values will prevent leaves with very few samples
4. Use subsampling, which is when only a fraction of samples used for each tree, by using only a fraction of the sample to grow each tree, it introduces randomess to prevent overfitting
5. Add alpha, which is L1 regularization
6. Add lambda, which is L2 regularization

The best model from the GridSearchCV using the XGBoost model produced the following performance metrics:

   | Metric         |     Result    |
   |:---------------|--------------:|
   | train RMSE     |        $8,059 |
   | val RMSE       |       $26,683 |
   | variance       |       $18,624 |
   | log val RMSE   |        0.1290 |

- This model yielded the highest variance, which is not ideal. However, it produced the best results in all the other metrics. Despite the larger variance and strong overfitting, it still performed better than all the other models on val. This suggests that this is the best model for predicting house prices in the real world.


### Best Model & Kaggle Submission

My best performing model used XGBoost with the best parameters selected through GridSearchCV. Using this model, I predicted the home prices for the test dataset from Kaggle and submitted it to the Leaderboard. My predictions obtains a final log RMSE of 0.12668, which earned me 541th place out of 3736 submissions (as of Mar 9th, 2025). 

![kaggle_submission_2](https://github.com/user-attachments/assets/f3f4aaa9-8142-4bfa-be86-388ee88cc3aa)


# Model Analysis

### From Lasso, RandomForest and XGBoost models, I plotted the features with the highest importance. 

##### Lasso:
![importance_lasso](https://github.com/user-attachments/assets/dcce2a9b-f164-4736-a9b5-0a8134ecdb68)

##### RandomForest:
![importance_rf](https://github.com/user-attachments/assets/bc2151b8-78e4-4839-9a6d-bf56ae8e6878)

##### XGBoost:
![importance_xgb](https://github.com/user-attachments/assets/f65e8b04-81af-4ec6-ab64-9173380130da)

### Observations: From the above plots, it is interesting to see which features were deemed to be important in all 3 models, and which ones only appeared in 1 or 2 models. 

1. OverallQual scored first on both of the ensemble models. This is in line with the original Pearson's correlation coefficient, which also awarded the highest r value to OverallQual. It did not score as high on Lasso, but still the 8th largest coefficient.
2. GrLivArea has the largest coefficient by far in the Lasso model, which gave it an coefficient of 0.3096, which is more than triple the the coefficient of the next feature down, Neighborhood_StoneBr with 0.09029. It ranked second on the RF model, and 9th on the XGBoost model. This reaffirms my initial observations that the size of the home highly correlated with the sale price.
3. Some of my engineered features were found to be important by the models. HouseAge is the 4th most important in the RF model, and TotalBaths 8th. The XGB model also valued TotalBaths at 8th most importance, but it deemed FireBedRatio to be even more telling! FireBedRatio came in as the 7th most important feature according to XGBoost.
4. The Lasso model seemed to emphasize the various neighborhoods more, with the 2nd, 4th and 5th features all being specific neighborhoods. This makes sense though, as the initial correlation also showed Neighborhood having the second highest correlation coefficient to SalePrice. To the contrary, the ensemble models did not assign as high of an importance to any of the neighborhood, they are further down the list.


# Conclusion & Next Steps

I believe that my XGBoost model will serve the Ames City Assesor well in their job of valuing homes in Ames City at their market value. However, there is still plenty of room to improve the model. My next steps include:

1. Obtain more data, especially more recent data, as this is key to keeping the model up to date with current housing trends.
2. Capture more quality-related data, such as the quality of the bathrooms, ranked by an observer on a scale of 1 to 10.
3. Obtain and analyze pictures of the home. Exterior visuals should be the easiest to obtain by using services such as Google Maps, then, use multimodal LLM to analyze pictures for patterns that correlate to the SalePrice. For example, a street where more cars park on the street could correspond to lower sales prices than a different street where most residents parks in a driveway, as parking on the street could indicate lower income areas, and/or higher density living. 
4. Include additional home features that may indicate investment in the home, such as electric car chargers and solar panels.
5. Incorporate other estimators to boost accuracy of my model. Online estimators such as Zillow's Zestimate or Redfin, offer their own models to help the public estimate home prices. Similar to the concept of gradient boosting, we can utilize these publically available estimates to improve our own estimates for Ames City. However, before doing this, I will need to research both websites to ensure using their estimates in our model does not violate any of their policies.


# References

##### Reference 1 : 
Realtor.com: ["How Much Does it Cost to Add a Bathroom, Bedroom, or More Room to Your Home?"](https://www.realtor.com/advice/home-improvement/how-much-does-it-cost-to-add-a-bathroom/)
##### Reference 2
Angi.com: ["Does Installing a Fireplace Increase the Value of Your Home?"](https://www.angi.com/articles/do-fireplaces-make-your-home-value-hot.htm)
##### Reference 3
IAstate.edu: ["How cold does it get in Ames in winter?](https://www.stat.iastate.edu/life-ames-faq)




