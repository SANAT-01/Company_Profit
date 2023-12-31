# -*- coding: utf-8 -*-
"""Exposys.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LUPzk6o6e3XnZaRu-ijEa17rL6rjplSO

## Importing required libraries
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RANSACRegressor

"""## Initialize the models"""

linear_reg = LinearRegression()
svr = SVR(kernel='linear')
decision_tree_reg = DecisionTreeRegressor()
random_forest_reg = RandomForestRegressor(n_estimators=50)
gradient_boosting_reg = GradientBoostingRegressor()
poly_reg = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])

"""## Loading data"""

data = pd.read_csv('50_Startups.csv')

data.head(), data.shape

data.dtypes

data.isnull().sum()

data.describe()

"""Here, we can that the min contains zero val in R&D Spend and Marketing spend which should not be zero there

## Data Preprocessing

Imputing missing values, with the average over the same data
"""

# Calculate the mean of "Administration" excluding zero values
mean1 = data[data['R&D Spend'] != 0]['R&D Spend'].mean()
mean2 = data[data["Marketing Spend"] != 0]['Marketing Spend'].mean()

# Impute zero values with the calculated mean
data.loc[data['R&D Spend'] == 0, 'R&D Spend'] = mean1
data.loc[data['Marketing Spend'] == 0, 'Marketing Spend'] = mean2

data.describe()

"""Removal of outliers"""

sns.boxplot(data["R&D Spend"])

sns.boxplot(data["Administration"])

sns.boxplot(data["Marketing Spend"])

sns.boxplot(data["Profit"])

"""Here, all of the data looks fine. So no need to remove the outliers"""

Main_data = data.copy()

"""## Linear Regression

OLS Assumptions
"""

cols = data.columns
cols

plot_data = data[cols]
plot_data.hist(bins=20)

"""Here, the data is normally distributed. Great !!

Linearity and Homoscacidity
"""

# Determine the number of rows and columns for the grid
r = 2
c = 2
fig, axes = plt.subplots(r, c, figsize=(8, 8))

# Iterate over the columns and plot scatter plots
for i, column in enumerate(cols):
    row = i // c  # Calculate the row index
    col = i % c  # Calculate the column index
    axes[row, col].scatter(data[column], data['Profit'])
    axes[row, col].set_xlabel(column)
    axes[row, col].set_ylabel('Target')
    axes[row, col].set_title(f'Scatter plot: {column} vs Profit')

# Adjust the layout and spacing
plt.tight_layout()

# Display the plot
plt.show()

"""Here the independent variables are also linear

Multicollinearity
"""

cols = data.columns[:3]
# cols.remove(3)
cols

from statsmodels.stats.outliers_influence import variance_inflation_factor

# To make this as easy as possible to use, we declare a variable where we put
# all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
variables = data[cols]

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = variables.columns

vif

"""Since, VIF of R&D Spend and Makrjeting spend are below 10. So we will accept it"""

plt.figure(figsize=(5,5))
sns.heatmap(data.corr(),annot=True,cmap='Oranges_r')
plt.show()

"""Feature selection

Dividing my data into train and test set

Since, there is a bell-curve in my data so standardization is more preferable.  normalization will compress these values into a small range.
"""

X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

from sklearn.preprocessing import StandardScaler

# Normalize the data
sc = StandardScaler()
X = sc.fit_transform(X)

X = pd.DataFrame(columns = ['R&D Spend', 'Administration', 'Marketing Spend'], data = X)

from sklearn.model_selection import train_test_split

# Split the data into train and test sets
# Split the data into features (X) and target variable (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

len(X_train)

# Functions that plots the actual versus predicted values
def plottings(actual_values, predicted_values):
    plt.scatter(actual_values, predicted_values)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')

    # Add a line of best fit or regression line
    line_of_best_fit = np.polyfit(actual_values, predicted_values, 1)
    plt.plot(actual_values, np.polyval(line_of_best_fit, actual_values), color='red')

    plt.show()

"""Creating a K cross fold validation function to select best model"""

def k_cross_fold(x_train, y_train, model, k):
  Rmse = float("inf")
  pre = None
  intercept = None
  prev = None
  coeff = None
  lg = len(x_train)
  split_amount = int(lg/k)
  best_model = None
  for i in range(k):
      if i == 0:
        train_x = x_train.iloc[split_amount*(i+1) : ]
        train_y = y_train[split_amount*(i+1) : ]
        test_x = x_train.iloc[ : split_amount*(i+1)]
        test_y = y_train[ : split_amount*(i+1)]

      elif i == k-1:
        train_x = x_train.iloc[ : split_amount*(i)]
        train_y = y_train[ : split_amount*(i)]
        test_x = x_train.iloc[split_amount*(i) : ]
        test_y = y_train[split_amount*(i) : ]

      else :
        test_x = x_train.iloc[int(split_amount* (i)) : int(split_amount* (i+1))]
        test_y =  y_train[int(split_amount* (i)) : int(split_amount* (i+1))]
        train_x = x_train.iloc[ : int(split_amount* (i))]
        train_x = np.append(train_x, x_train.iloc[int(split_amount* (i+1)) : ], axis = 0)
        train_y = y_train[ : int(split_amount* (i))]
        train_y = np.append(train_y, y_train[int(split_amount* (i+1)) : ], axis = 0)
#       print(train_x.shape)
      model.fit(np.array(train_x),train_y)
      y_hat = model.predict(test_x)
      coeff = model.coef_
      intercept = model.intercept_
      print('MAPE on train data: ', mean_squared_error(test_y, y_hat)**(0.5))

      if Rmse > mean_squared_error(test_y, y_hat)**(0.5):
        Rmse = mean_squared_error(test_y, y_hat)**(0.5)
        pre = intercept
        prev = coeff
        best_model = model
        print(Rmse)

  return Rmse, pre, prev, best_model

df = pd.DataFrame(columns=['Regression Algorithm', 'RMSE', 'MAE', 'R2'])

# Fit the models on the training data
linear_reg = k_cross_fold(X_train, y_train, linear_reg, 10)[-1]
# Make predictions on the test set
linear_reg_preds = linear_reg.predict(X_test)
linear_reg_rmse = mean_squared_error(y_test, linear_reg_preds, squared=False)
linear_reg_mae = mean_absolute_error(y_test, linear_reg_preds)
linear_reg_r2 = r2_score(y_test, linear_reg_preds)

plottings(y_test, linear_reg_preds)

print('Linear Regression RMSE:', linear_reg_rmse)
print('Linear Regression MAE:', linear_reg_mae)
print('Linear Regression R2 Score:', linear_reg_r2)

df.loc[0] = ['Linear', linear_reg_rmse, linear_reg_mae, linear_reg_r2]

"""## POlynomial Regression"""

poly_reg.fit(X_train, y_train)
poly_preds = poly_reg.predict(X_test)
poly_rmse = mean_squared_error(y_test, poly_preds, squared=False)
poly_mae = mean_absolute_error(y_test, poly_preds)
poly_r2 = r2_score(y_test, poly_preds)

plottings(y_test, poly_preds)

print('Polynomial RMSE:', poly_rmse)
print('Polynomial MAE:', poly_mae)
print("Polynomial R2", poly_r2)

df.loc[4] = ['Polynomial', poly_rmse, poly_mae, poly_r2]

"""## SVR (Support Vector Regression)

For support Vector Machine we need to Standardize both the dependent and independent variable
"""

Main_data.head()

X = Main_data[["R&D Spend", "Administration", "Marketing Spend"]]
y = Main_data["Profit"]

"""The y variable must be converted into suitable format"""

y = np.array(y).reshape(50,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

svr.fit(X_train, y_train)
svr_preds = svr.predict(X_test)

actual = sc_y.inverse_transform(y_test).reshape(1,8)[0]
predicted = sc_y.inverse_transform(svr_preds.reshape(8,1)).reshape(1,8)[0]
svr_rmse = mean_squared_error(actual, predicted, squared=False)
svr_mae = mean_absolute_error(actual, predicted)
svr_r2 = r2_score(actual, predicted)

plottings(actual, predicted)

print('SVR RMSE:', svr_rmse)
print('SVR MAE:', svr_mae)
print("SVR R2 Score", svr_r2)

df.loc[1] = ['SVR', svr_rmse, svr_mae, svr_r2]

"""We have tried with different Kernel, and the best we found was linear

## Decision Tree Regressor

Data Scaling or Feature scaling is not required
"""

X = Main_data[["R&D Spend", "Administration", "Marketing Spend"]]
y = Main_data["Profit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

decision_tree_reg.fit(X_train, y_train)
decision_tree_preds = decision_tree_reg.predict(X_test)
decision_tree_rmse = mean_squared_error(y_test, decision_tree_preds, squared=False)
decision_tree_mae = mean_absolute_error(y_test, decision_tree_preds)
decision_tree_r2 = r2_score(y_test, decision_tree_preds)

plottings(y_test, decision_tree_preds)

print('Decision Tree RMSE:', decision_tree_rmse)
print('Decision Tree MAE:', decision_tree_mae)
print('Decision Tree R2 Score:', decision_tree_r2)

df.loc[1] = ['Decision Tree', decision_tree_rmse, decision_tree_mae, decision_tree_r2]

"""## Random Forest Regressor"""

random_forest_reg.fit(X, y)
random_forest_preds = random_forest_reg.predict(X_test)
random_forest_rmse = mean_squared_error(y_test, random_forest_preds, squared=False)
random_forest_mae = mean_absolute_error(y_test, random_forest_preds)
random_forest_r2 = r2_score(y_test, random_forest_preds)

plottings(y_test, random_forest_preds)

print('Random Forest RMSE:', random_forest_rmse)
print('Random Forest MAE:', random_forest_mae)
print('Random Forest R2 Score:', random_forest_r2)

df.loc[2] = ['Random Forest', random_forest_rmse, random_forest_mae, random_forest_r2]

"""## Graadient Boosting"""

gradient_boosting_reg.fit(X_train, y_train)
gradient_boosting_preds = gradient_boosting_reg.predict(X_test)
gradient_boosting_rmse = mean_squared_error(y_test, gradient_boosting_preds, squared=False)
gradient_boosting_mae = mean_absolute_error(y_test, gradient_boosting_preds)
gradient_boosting_r2 = r2_score(y_test, gradient_boosting_preds)

plottings(y_test, gradient_boosting_preds)

print('Gradient Boosting RMSE:', gradient_boosting_rmse)
print('Gradient Boosting MAE:', gradient_boosting_mae)
print("Gradient Boosting R2", gradient_boosting_r2)

df.loc[3] = ['Gradient Boosting', gradient_boosting_rmse, gradient_boosting_mae, gradient_boosting_r2]

"""## Choosing the best model"""

df = df.sort_values(by= ['RMSE', 'MAE', 'R2'])
df

"""Saving the best model, which is Gradient Boosting"""

from joblib import Parallel, delayed
import joblib
import pickle

# Save the model as a pickle in a file
joblib.dump(random_forest_reg, 'Best_model.pkl')
# pickle.dump(scaler, open('Scaler.pkl', "wb"), protocol=0)

# Load the model from the file
Model = joblib.load('Best_model.pkl')
# Scaler = pickle.load(open('Scaler.pkl', 'rb'))
# x = Scaler.transform([data.iloc[40][:3]])
# Use the loaded model to make predictions

x = [X.iloc[40][:3]]
Model.predict(x)

[X.iloc[40][:3]], y[40]

