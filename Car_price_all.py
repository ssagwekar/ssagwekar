#importing and file reading
import pandas as pd
car_price = pd.read_csv('CarPrice_Assignment.csv')

car_price.dtypes
car_price.info()     #no null values found

# Display descriptions for each feature.
descriptions = {
    'car_ID': "Unique identifier for each car",
    'symboling': "Insurance risk rating for the car",
    'CarName': "Name or model of the car",
    'fueltype': "Type of fuel used (gas or diesel)",
    'aspiration': "Method of air intake for the engine (std or turbo)",
    'doornumber': "Number of doors on the car",
    'carbody': "Type of car body or design (sedan, hatchback, etc.)",
    'drivewheel': "Type of drivetrain or wheels (FWD, RWD, 4WD)",
    'enginelocation': "Engine location (front or rear)",
    'wheelbase': "Distance between front and rear axles",
    'carlength': "Length of the car",
    'carwidth': "Width of the car",
    'carheight': "Height of the car",
    'curbweight': "Weight of the car without passengers or cargo",
    'enginetype': "Type of engine (ohc, ohcv, etc.)",
    'cylindernumber': "Number of cylinders in the engine",
    'enginesize': "Size of the engine (in cc or ci)",
    'fuelsystem': "Type of fuel injection system (mpfi, etc.)",
    'boreratio': "Ratio of cylinder bore diameter to stroke length",
    'stroke': "Length of the engine stroke",
    'compressionratio': "Engine compression ratio",
    'horsepower': "Engine power output (in hp)",
    'peakrpm': "Engine's peak RPM for generating power",
    'citympg': "Fuel efficiency in city driving (mpg)",
    'highwaympg': "Fuel efficiency on the highway (mpg)",
    'price': "Price of the car"
}

for i in car_price.columns:
    print(f"{i}: {descriptions.get(i, 'No description available')}")

data = car_price.copy()
#dropping noisy and irrelevant data
data1 = data.drop('car_ID',axis=1)

#checking top 10 values
data1.head(10)
data1.tail(10)

#checking for duplicate values if any
print("Duplicate values = ",data1.duplicated().sum())

#checking for null values
print("Null values =\n", data1.isnull().sum(axis=0),
      "\n------------------------------",
      "\nNO NULL VALUES FOUND ")

#descriptive stats finding manually
numerical = []
for i in data1.columns:
    if data1[i].dtype in [int, float, 'int64','float64']:
        numerical.append(i)

category =[]
for i in data1.columns:
    if data1[i].dtype in ['object']:
        category.append(i)
        
        
'''num_col = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 
              'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
              'citympg', 'highwaympg', 'price']'''

import statistics as st
from scipy.stats import mode
import scipy.stats as stat
for column in numerical:
    data_column = data1[column]             #extraxt columns from dataframe for operations
    print(f"\nStatistic for '{column}':")
    print(f"Count: {data_column.count()}")  #from pandas
    print(f"Mean: {st.mean(data_column)}")  
    print(f"Median: {st.median(data_column)}")
    print(f"Mode: {mode(data_column,keepdims = 'TRUE').mode[0]}")  #from scipy
    print(f"Variance: {st.variance(data_column)}")
    print(f"Standard Deviation: {st.stdev(data_column)}")
    print(f"Minimum: {data_column.min()}")  #from pandas
    print(f"25th Percentile: {data_column.quantile(0.25)}")
    print(f"50th Percentile (Median): {data_column.median()}")
    print(f"75th Percentile: {data_column.quantile(0.75)}")
    print(f"Maximum: {data_column.max()}")
    
for column in category:
    data_column = data1[column]
    print(f"\nStatistic for {column} :")
    print(f"Count: {data_column.count()}")
    print(f"Unique: {data_column.nunique()}")
    print(f"Top: {data_column.mode().iloc[0]}")
    print(f"Mode: {data_column.value_counts().iloc[0]}")
    

#direct method to check descriptive statistic
stats_num = data1.describe()
stats_cat = data1.describe(include = 'object')

#carname has 147 unique values need to drop or split to company name
data1.insert(1 ,'Company',data1['CarName'].str.split(' ').str[0])
data2 = data1.drop('CarName',axis=1)
stats_cat = data2.describe(include = 'object')

data2['Company'].unique()
'''Out[10]: 
array(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'maxda', 'mazda', 'buick', 'mercury',
       'mitsubishi', 'Nissan', 'nissan', 'peugeot', 'plymouth', 'porsche',
       'porcshce', 'renault', 'saab', 'subaru', 'toyota', 'toyouta',
       'vokswagen', 'volkswagen', 'vw', 'volvo'], dtype=object)'''

#replacing names with correct ones which are mistaken

brand_name = {
    'maxda'     : 'mazda',
    'Nissan'    : 'nissan',
    'porcshce'  : 'porsche',
    'toyouta'   : 'toyota',
    'vokswagen' : 'volkswagen',
    'vw'        : 'volkswagen' 
    }
data2['Company'] = data2['Company'].replace(brand_name)
data2['Company'].unique()

'''data2.to_csv("C://Users//ACER//Downloads//Car_price_new.csv",index = False)'''

#Visualization

num_col = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 
              'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
              'citympg', 'highwaympg', 'price']

cat_col = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
           'enginetype', 'cylindernumber', 'fuelsystem']

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Company V/s Average price
avg_price = data2.groupby('Company')['price'].mean().reset_index()
plt.bar(avg_price['Company'], avg_price['price'],align= 'center')
plt.xlabel('Brand name')
plt.ylabel('Average Prices')
plt.xticks(rotation=45)
plt.title('Average Car Prices by Company')
plt.show()

#using seaborn 
#Fuel type V/s Average price
avg_price = data2.groupby('fueltype')['price'].mean().reset_index()
sns.set(style="whitegrid")
sns.barplot(x='fueltype', y='price', data=avg_price,width=0.4)
plt.xlabel('Fuel Type')
plt.ylabel('Average Prices')
plt.xticks(rotation=45)
plt.title('Average Car Prices by fuel type')
plt.show()

#correlation matrix
before_corr = data2.corr(numeric_only='TRUE')
plt.figure(figsize=(12, 10))
sns.heatmap(before_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

#Comparing price with categorical columns
fig, axes = plt.subplots(1, len(cat_col), figsize=(40, 6))  # Adjust the figsize as needed

for i, col in enumerate(cat_col):
    sns.stripplot(x=col, y='price', data=data2, size=4, ax=axes[i])
    axes[i].set_title(f"Price V/S {col}")
    axes[i].set_xlabel(f"{col}")
    axes[i].set_ylabel('Price')

plt.tight_layout()
plt.show()

'''Create Subplots:

1.plt.subplots(1, len(cat_col), figsize=(40, 6)): This part is like saying, "I want one row (1),
 and each category in my list should have its own subplot." The len(cat_col) part ensures that
 you'll have as many subplots as there are categories in your list.

2.Unpack Result:

fig, axes = ...: This part is like saying, "I want to keep track of the whole figure (fig), 
and each subplot separately (axes)." The axes variable becomes an array where axes[0] 
corresponds to the subplot for category A, axes[1] for category B, and axes[2] for category C.
'''
    
#Comparing price with numerical columns 
num_col = ['symboling','carlength', 'carwidth','enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
              'citympg', 'highwaympg',]

fig ,axes = plt.subplots(2,5,figsize=(16,8))
axes = axes.flatten() # Flatten the 2D array of subplots into a 1D array
for i,col in enumerate(num_col):
    sns.scatterplot(x=col ,y='price',data=data2 ,size=4 ,ax=axes[i])
    axes[i].set_title(f"Price V/S {col}")
    axes[i].set_xlabel(f"{col}")
    axes[i].set_ylabel('Price')
   
plt.tight_layout()
plt.show()

#converting categorical columns to one-hot
data3 = pd.get_dummies(data2,drop_first='TRUE')

#checking Normal distribution curve using sns.histplot and kernel density estimate
# sorting all coulmns for Q_Q plot and shapiro test

df = data3.copy()
df_num = df.iloc[:,0:15]
num_col = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
       'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']

#plotting histogram and Q-Q plot

for i in df_num.columns:
    df_num[i] = df_num[i].sort_values().values

#1. HISTOGRAM (Normal distribution curve)
fig,axes = plt.subplots(len(num_col),2,figsize=(15, 5 * len(num_col)))
axes = axes.flatten()
for i,col in enumerate(num_col):
    sns.histplot(data=df_num[col],kde='TRUE',ax=axes[i*2])
    axes[i*2].set_title(f"Histogram plot for {col}")
    axes[i*2].set_xlabel(f"Values")
    axes[i*2].set_ylabel(f"{col}")
 
#2.Q-Q plot
for i,col in enumerate(num_col):
    stat.probplot(df_num[col],dist='norm',plot=axes[(2*i)+1])
    axes[(2*i)+1].set_title(f"Q-Q plot for {col}")
    axes[(2*i)+1].set_xlabel(f"Imaginary values")
    axes[(2*i)+1].set_ylabel(f"{col}")
    
axes[(2*i)+1].axis('off')  #used to turn off the axis 

plt.tight_layout()
plt.show()

#3.Shapiro wilk test
for i in df_num.columns:
     shapiro_stats, p_value = stat.shapiro(df_num[i])
     print(f"\nShapiro Wilk-Test for {i}:\nStatisrtics : {shapiro_stats}\np_value : {p_value}")
     
     if p_value > 0.05:
         print(f"/nData in {i} follows Normal Distribution\n")
     else:
        print(f"/nData in {i} does not follows Normal Distribution\n")
        
'''CONCLUSION - Thus df_num numeric data does not follows the normal distribution hence using 
                minmax_scale() for normalization of data'''
                
#normalizing numeric data with minmax() scale

from sklearn.preprocessing import minmax_scale
data4 = minmax_scale(data3)  #O/P will be in numpy.ndarray

#converting numpy.ndarray to DataFrame using pandas
data4 = pd.DataFrame(data4,columns=data3.columns)

filtered_column = data4.iloc[:,0:15]
correlation = filtered_column.corr(numeric_only='TRUE')
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

#correlation.to_csv("Downloads\\Corelation_matrix.csv",index = False)

'''
data4 = data4.drop('enginelocation_rear',axis=1) 
Need to drop enginelocation_rear column from data4 because it has 2category of Rear=202 and front=3,
while stratistication , it would not take minimum value'''
#Spliting values in input and output
Y = data4[['price']]
X = data4.drop(['price'],axis =1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=1234)

'''Since ‘price’ is a continuous variable, it’s unusual to use stratification. 
Stratification is typically used for categorical/classification problems where the target 
variable is categorical.
'''

#data4.to_csv('F:\\BIA\\Project to do\\archive\\normalized.csv',index='FALSE')

#import linear regression 
from sklearn.linear_model import LinearRegression,Lasso,Ridge
li_reg = LinearRegression()

#model training
li_reg.fit(X_train,Y_train)

#model prediction
Y_li_pred = li_reg.predict(X_test)

#slope and intercept
slope_li_reg = li_reg.coef_
intercept_li_reg = li_reg.intercept_

#checking for errors in model
from sklearn.metrics import mean_squared_error
import math as mt

RMSE_li_reg = mt.sqrt(mean_squared_error(Y_test,Y_li_pred))
print(f"Root Mean Square for linear regression= {RMSE_li_reg}")                 #392504951.5662341 

#MAE
from sklearn.metrics import mean_absolute_error
MAE_li_reg = mean_absolute_error(Y_test,Y_li_pred)  #61298974.901960544
print(f"Mean Absolute Error for linear regression= {MAE_li_reg}")    

#R-square
score_li_reg = li_reg.score(X_test,Y_test) #-8.777895174123062e+18
print(f"R-square for linear regression= {score_li_reg}")  


#import lasso regression
lasso = Lasso(alpha=0.0002774130705900127)
lasso.fit(X_train,Y_train)
Y_lasso_pred = lasso.predict(X_test)

#slope and intercept
slope_lasso_reg = lasso.coef_
intercept_lasso_reg = lasso.intercept_

nonzero_coefficients = np.sum(lasso.coef_ != 0)
print(f"Number of non-zero coefficients: {nonzero_coefficients}")

print("Lasso Coefficients:")
for feature, coef in zip(X.columns, lasso.coef_):
    print(f"{feature}: {coef}")

#RMSE
RMSE_lasso_reg = mt.sqrt(mean_squared_error(Y_test,Y_lasso_pred))
print(f"Root Mean Square for lasso regression= {RMSE_lasso_reg}")               #

#MAE
from sklearn.metrics import mean_absolute_error
MAE_lasso_reg = mean_absolute_error(Y_test,Y_lasso_pred)  #
print(f"Mean Absolute Error for lasso regression= {MAE_lasso_reg}")

#R-square
score_lasso_reg = lasso.score(X_test,Y_test) #
print(f"R-square for lasso regression= {score_lasso_reg}") 


'''
CROSS_VALIDATION FOR LASSO TO GET OPTIMUM VALUE OF APLHA
from sklearn.linear_model import LassoCV
 
lasso_cv = LassoCV(cv=20)
lasso_cv.fit(X_train, Y_train)
pred = lasso_cv.predict(X_test)
best_alpha = lasso_cv.alpha_

nonzero_coefficients = np.sum(lasso_cv.coef_ != 0)
print(f"Number of non-zero coefficients: {nonzero_coefficients}")'''

#import Ridge regression
ridge = Ridge(alpha=0.8)
ridge.fit(X_train,Y_train)
Y_ridge_pred = ridge.predict(X_test)

#slope and intercept
slope_ridge_reg = ridge.coef_
intercept_ridge_reg = ridge.intercept_

nonzero_coefficients = np.sum(ridge.coef_ != 0)
print(f"Number of non-zero coefficients: {nonzero_coefficients}")

print("Ridge Coefficients:")
for feature, coef in zip(X.columns, ridge.coef_):
    print(f"{feature}: {coef}")

#RMSE
RMSE_ridge_reg = mt.sqrt(mean_squared_error(Y_test,Y_ridge_pred))
print(f"Root Mean Square for ridge regression= {RMSE_ridge_reg}")               #

#MAE
from sklearn.metrics import mean_absolute_error
MAE_ridge_reg = mean_absolute_error(Y_test,Y_ridge_pred)  #
print(f"Mean Absolute Error for ridge regression= {MAE_ridge_reg}")

#R-square
score_ridge_reg = ridge.score(X_test,Y_test) #
print(f"R-square for ridge regression= {score_ridge_reg}")

#RainForest Regressor
#Ensemble learning
from sklearn.ensemble import RandomForestRegressor
import numpy as np

rfc = RandomForestRegressor(random_state=1234)

# Training
rfc.fit(X_train, np.ravel(Y_train.values))

# Prediction
Y_rfc_pred = rfc.predict(X_test)

#RMSE
RMSE_rfc_reg = mt.sqrt(mean_squared_error(Y_test,Y_rfc_pred))
print(f"Root Mean Square for rfc regression= {RMSE_rfc_reg}") #4.4%

#MAE
from sklearn.metrics import mean_absolute_error
MAE_rfc_reg = mean_absolute_error(Y_test,Y_rfc_pred)  #3.1%
print(f"Mean Absolute Error for rfc regression= {MAE_rfc_reg}")

#R-square
score_rfc_reg = rfc.score(X_test,Y_test) #88.9%
print(f"R-square for rfc regression= {score_rfc_reg}")



            

       


    








    
    













