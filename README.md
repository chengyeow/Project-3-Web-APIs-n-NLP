# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Ames Housing Data and Kaggle Challenge


### Problem Statement

To help novice property flippers manage risks, we will provide consultation upon application, based on our home valuation that predicts the sale prices of homes in Ames, Iowa. We have modeled Linear Regression, Ridge, Lasso and Elastic Net models. The sale price prediction software is also helpful for property buyers or sellers.

To better manage risks for new property flippers, we recommend the Reno Flip.

Thus, our recommendations will focus on property features that are more likely to increase property value when renovated, and which features to de-prioritise. It is important to have this modelling predicts a good accuracy of the features as it will help the audience (property flippers) to maximise the property potential.

Success will be driven by the most optimal linear regression models with the best RMSE score using Linear regression, ridge, lasso and elastic net modelling.


### Executive Summary

Using the Ames Housing Dataset that is available on Kaggle, we want to identify which features are the best predictors of housing price and create a linear regression model that will help us make predictions with the best R2 score and also the RMSE score.

The Kaggle challenge offers a train.csv for us to to train our model with, and a test.csv which we will fit the model onto, in order to make our predictions. Both train.csv and test.csv will be cleaned and performed feature engineering first and saved to train_clean.csv and test_clean.csv. The csv of the predictions is then uploaded to the Kaggle challenge for scoring.

The model will be tuned closely to the Ames Housing dataset, and we might be able to use our findings from the process to understand what are some key predictors we can use in predicting prices for Ames houses in the United States. However, given that this data set has some features that are very specific to Ames, Iowa (e.g. neighborhood), it will not be perfect fit for other housing data in the U.S.

The model, and our understanding of the key features will be beneficial to existing home owners who might be considering selling their property to have a gauge of what prices their property could fetch. We will also advise the home owners what specific features can be upgraded to help them fetch a higher selling price.

With the right kind of predictors, this model aims to make the property market in Ames a fair and competitive one that will benefit both buyers and sellers.

Please note that there are two files to this project.
- G2 Ames Housing Data Cleaning - Part 1
- G2 Ames Housing Modeling - Part 2

### Datasets
* [`train.csv`](./datasets/train.csv): Train set for training the machine learning models.
* [`test.csv`](./datasets/test.csv): Test set provided for Kaggle submission.
* [`train_clean.csv`](./datasets/train_clean.csv): Train set cleaned for modeling evaluation.
* [`test_clean.csv`](./datasets/test_clean.csv): Test set cleaned for Kaggle submission.


### Data Dictionary

* [`datadocumentation.pdf`](./datasets/datadocumentation.pdf): Data Dictionary used in train.csv and test.csv
* [`data_dictionary.csv`](./datasets/data_dictionary.csv): Dictionary used in train_clean.csv and test_clean.csv


### Data Import Cleaning
* There are a total of 2051 rows and 81 columns for train set and 878 columns and 80 columns for test set.

Firstly, need to idenify the columns with missing data. For each of the data, the relations were gathered for the commonality features. Missing values for the following features were filled with zero or NA except for the rows mentioned.

1) Mas Vnr Type
2) Basement features (An odd missing value in BasementFin Type 2 was filled with next highest average (Rec) as there is no relation to the other basement features.
3) Garage feaures (An garage item that has a detched garage and an area of 360 sf, had its missing data filled with average data)
4) Electrical data only has missing value in test set and filled with average value as every household should have the basic electrical feature.
5) Lot Frontage, this feature is important although there are missing value for train - 330 and test 160 missing values. To impute a reasonable value for ot Frontage, I used the 24 neighborhoods as the main reference and further breakdown down each neighborhood into 5 lot config (Corner, CulDSac, FR2, FR3, Corner).
6) Fireplace
7) Pool QC
8) Misc Feature, Alley and Fence, There is no significant relation for Fence and also with the Neighborhood. And the missing data is >80%. For Misc Features and Alley, the missing values is >90%. With the high number of missing data, it does not aid in the machine learning with more than 80% missing. Hence, these 3 features are dropped.

### Converting Ordinal to Numeric values
Replace the ordinal columns with numeric sequencial values for modelling evaluation.
Lot Shape, Utilities, Land Slope, Overall Qual, Overall Cond, Exter Qual, Exter Cond, Bsmt Qual, Bsmt Cond, Bsmt Exposure, BsmtFin Type 1, BsmtFin Type 2, HeatingQC, Electrical, KitchenQual, Functional,FireplaceQu, Garage Finish, Garage Qual, Garage Cond, Paved Drive, Pool QC.


### Exploratory Data Analysis

In this section, I did feature engineering on the features.
1) On the neighborhood feature, I classified the neighborhood into a list of order. It is done using series of other house features summation and median  it. Using the median and further grouped the neighborhood into 4 groups (7 neighborhood per group)
2) Combine condition 1 & 2 into positive numerical values.
3) Sale type order is classified further is the sale price can differ from new building or contract.
4) Outliners were removed for those standard deviation of 3 or above.
5) Age was built to see if there is any effect with the sale price. At age 0 for new houses, the price are strong. The other view is that as the house age increases, the sale price decreases.
6) Combine basement type 1 & 2 into single feature as they are in common.
7) Combine 1st and 2nd floor size into single feature as they are in common.
8) Combine half and full bath into single feature of total bath rooms.
9) Identify feature of room size by getting ground living area / total rooms above ground.
10) Combine all types of porch area into single feature, outside porch sf.
11) Months sold are checked and no correlaton with sale price.
12) Masonary vnr type and ms zoning looked into to further classified under the quality and the residentail type respectively.
13) Combine exterior 1 & 2 into single feature.
14) Classify house style numeric order and then dummified it.
15) Roof style is dummified and to keep the hip roof as it is classified as luxury roof.
16) Buildng type is dummified as there is no specific relation to the numeric values.


Outliners
Outliners will stew the modeling and the prediction. Hence it is important to clear the dataset for normal distribution.
1) Lot Frontage >250 removed.
2) Basement type 1 & 2 > 20,000 removed.
3) Total bath rooms >7 removed

Modeling Evaluation
After cleaning and feature engineering, the variable features are 33 to support the modeling prediction. Lasso regression would be the best regression to use as it is able to further reduce the coefficient for the less useful features to zero.

The mean RMSE was used as the baseline for the modeling at 68,815.11.

Explored various models, Linear, Ridge, Lasso and also using Pipeline Gridsearch and Elastic Net to find the best performed model

Scaling is necessary, given that we have a very mixed bag of features that are ordinal (typically 0 to 5), stated in age years, square feet sizes, etc.
StandardScaler was used for this purpose.

train_test_split was used to split the train set into train and test validation to train the model. It is used since those are industry standards that have been extensively researched.


### Conclusion and Recommendations

With all the modeling, based on the R2score, the train and test performed very close to each other. This indicates that the model are not too overfitting or underfitting. The R2 score are consistent for most of the models except LassoCV with polynomial features which brings out the best performance.

LassoCV with polynomial features regression by far has the best test score at 92.34%. With 33 features now, it is now more ideal and easy to explain to relevant stakeholders. Again, the model does not look to be overfitted.

LassoCV With Polynomial Features
- R^2 test Score: 0.9234
- Mean Root Square Error in dollars: 20,253.00

| Modeling                         | R^2 Score train | R^2 Score test | RMSE Score  |
|----------------------------------|-----------------|----------------|-------------|
| Linear Regression                |      0.8945     |     0.9039     |  21,692.69  |
| RidgeCV Regression               |      0.8944     |      0.904     |  22,673.49  |
| LassoCV Regression               |      0.8943     |     0.9039     |  22,685.97  |
|----- Pipeline and Gridsearch ----|-----------------|----------------|-------------|
| Ridge Regression                 |      0.8944     |      0.904     |  22,676.54  |
| Lasso Regression                 |      0.8944     |      0.904     |  22,682.82  |
| Lasso Regression with Polynomial |      0.9365     |     0.9234     |  20,253.00  |
| Elastic Net Regression           |      0.8925     |     0.9034     |  22,752.07  |

The same process was done with test set and it was submitted to Kaggle for scoring.

Kaggle scores using Lasso Regression with Polynomial. the model scored better in private and not so much in public.

Private: 19591.74119
Public: 26683.75808

Based on the LassoCV with Polynomial Features coefficients, these are the top 10 features that are useful. They have the strongest coefficient score.

1) rooms_above_size
2) single_story
3) basement_ceiling
4) outside_space
5) neighborhood_quality
6) local_features
7) fireplace_quality
8) paved_driveway
9) lot_frontage
10) floors_size

I would recommend upgrading the features that are easily upgraded for the home owners and it would fetch a higher selling price even after upgrading cost. Features such as room sizes will definitely fetch much higher price if we can expand the size. Other such as fireplace and kitchen quality are also worth upgrading as it would be easily achieved and yet fetch higher selling price.

It is important to note that the above conclusions are only based on the Ames housing dataset. More granular analysis is not possible as the data provided are actual sale prices of houses. In order to complement the above findings, more granular data such as buyer data and coordinates of home sales would be beneficial to delve deep into analysis such as buyer behaviour or neighbourhood studies. This could enable home owners or property developers to better target buyers, or to even educate buyers on the kind of houses they should look out for. With the right data, we will be able to more accurately predict and support the home buyers and sellers.
