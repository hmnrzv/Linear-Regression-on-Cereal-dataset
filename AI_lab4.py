from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

cereal_data=pd.read_csv(f'{PATH}/cereal_clean.csv')
cereal_data
PATH='/content/drive/MyDrive/5th semester/AI/LABS/LAB4'

#EXERCISE 1
C1=cereal_data.drop(columns=['rating','name']) #extracts column as a dataframe
print(type(C1))
C1
#Separating the target into variable Y1
Y1=cereal_data[['rating']]
print(type(Y1))
Y1
C1_train,C1_test,Y1_train,Y1_test = train_test_split(C1, Y1,test_size=0.2,random_state=0)
print('Size of original dataset (complete):',len(C1))
print('Size of train dataset (80%):',len(C1_train))
print('Size of test dataset (20%):',len(C1_test))

#EXERCISE 2
#Running multivariable linear regression
model = LinearRegression()
model.fit(C1_train, Y1_train)
y_all=model.predict(C1)
y_all
print('Training data r-square:', model.score(C1_train, Y1_train))
print('Test data r-square:', model.score(C1_test, Y1_test))
print('Intercept', model.intercept_)
#Regression coefficients for all features
coef_df = pd.DataFrame({'coef': model.coef_.flatten()}, index=C1_train.columns)
print(coef_df)
# Predict ratings for the test set
predicted_test = model.predict(C1_test)
# Create a copy of the test data to attach predictions
test_data = cereal_data.loc[C1_test.index].copy()
test_data['Predicted_Rating'] = predicted_test
# Find the cereal with the highest predicted rating
best_cereal = test_data.loc[test_data['Predicted_Rating'].idxmax()]

#EXERCISE 3
print("Best cereal on test data:")
print(best_cereal[['name', 'Predicted_Rating']])
new_row = pd.DataFrame([{'name': 'Chocolate Bread', 'calories': 49, 'protein': 3.5, 'fat': 0, 
                         'sodium': 138, 'fiber': 17, 'carbo': 10, 'sugars': 0.5, 'potass': 350, 
                         'vitamins': 10, 'weight': 1, 'rating': None}])
cereal_data = pd.concat([cereal_data, new_row], ignore_index=True)
cereal_data
# Select the last added row 
choc_cereal = cereal_data.tail(1)
# Select the same features used for training (exclude 'name' and 'rating')
features = choc_cereal.drop(['name', 'rating'], axis=1)
predicted_rating = model.predict(features)
print("Predicted rating for Chocolate Bread:", predicted_rating[0])
cereal_data.loc[cereal_data.index[-1], 'rating'] = predicted_rating[0]
