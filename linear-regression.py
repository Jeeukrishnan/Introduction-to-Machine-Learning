import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
import numpy as np
from sklearn.metrics import mean_squared_error   

'''
We can use Scikit learn to build a simple linear regression model
you can use it use it like 
model = LinearRegression() 
see the documentation here: 
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

To get Mean Squared Error (MSE) you can use 
mean_squared_error(Actual,Prediction)
'''


#### Your Code Goes Here #### 

## Step 1: Load Data from CSV File ####

dataframe=pd.read_csv("student_scores.csv")
print (dataframe) 
## Step 2: Plot the Data ####
X=dataframe["Hours"].values.reshape(-1,1)
Y=dataframe["Scores"].values.reshape(-1,1)
plt.plot(X,Y,'o')
plt.show()

## Step 3: Build a Linear Regression Model ####
X_train,Y_train=X[0:20] ,Y[0:20]
X_test,Y_test=X[19:],Y[19:]
print(X_train,X_test)
model=LinearRegression()
model.fit(X_train,Y_train)



## Step 4: Plot the Regression line ####
regressionline=model.predict(X)
plt.plot(X,regressionline)
plt.plot(X_test,Y_test,'o')
plt.plot(X_train,Y_train,'o')
plt.show()


## Step 5: Make Predictions on Test Data ####
y_predictions=model.predict(X_test)
print (y_predictions)
## Step 6: Estimate Error #### 
print(mean_squared_error(Y_test,y_predictions))

