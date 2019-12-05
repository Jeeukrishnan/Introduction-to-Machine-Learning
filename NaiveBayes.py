import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error

##load data
dataframe=pd.read_csv("train.csv")
print(dataframe.describe())
dataframe=dataframe.drop(["Name"],axis=1)
dataframe=dataframe.drop(["Sex"],axis=1)
dataframe=dataframe.drop(["Ticket"],axis=1)
dataframe=dataframe.drop(["Embarked"],axis=1)
dataframe=dataframe.drop(["Cabin"],axis=1)





##plot data

 
ages=dataframe["Age"].values.reshape(-1,1)
fares=dataframe["Fare"].values.reshape(-1,1)
survive=dataframe["Survived"].values.reshape(-1,1)
plt.plot(ages,fares,'o')
plt.show()

colors=[]
for item in survive:
      if item ==0:
          colors.append('red')
      else :
            colors.append('green')

plt.scatter(ages,fares,color=colors)
plt.show() 

##Build a NB Model
Features=dataframe.drop('Survived',axis=1).values.reshape(-1,1)
Targets=dataframe['Survived'].values.reshape(-1,1) 
Features_train,Targets_train=Features[0:710],Targets[0:710]
Features_test,Targets_test=Features[710:],Targets[710:]


model=GaussianNB() 
model.fit(Features_train,Targets_train)  

##print predicted vs Actuals
print(model.predict(Features_test))       
          