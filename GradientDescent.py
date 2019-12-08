import pandas as pd
import matplotlib.pyplot as plt  

#helper function
def plot_regression_line(X,m,b):
	regression_x = X.values
	regression_y = []
	for x in regression_x:
		y = m*x + b
		regression_y.append(y)

	plt.plot(regression_x,regression_y)
	


#read data
df=pd.read_csv("student_scores.csv")

#plot data
X=df["Hours"]
Y=df["Scores"]

plt.plot(X,Y,'o')

#def a function gradient descent such that it takes m,c and return a better value of m,c
##so that it reduces error

m=0
c=0
def grad_desc(X,Y,m,c):
     for point in zip(X,Y):
         x=point[0]
         y_actual=point[1]
         y_prediction=m*x+c
          
         error=y_prediction-y_actual
          
         delta_m= -1*(error*x)*0.23
         delta_c=-1*(error)*0.23      #0.23 is learning rate
         
         
         
         m=m+delta_m
         c=c+delta_c
         return m,c
     
#for i in range(0,10)
m,c=grad_desc(X,Y,m,c) 
print (m,c)
plot_regression_line(X,m,c)


plt.show()    