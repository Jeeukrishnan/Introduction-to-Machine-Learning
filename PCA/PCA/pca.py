from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv("Iris.csv")
print (df.head())

labels=df["Species"]

X=df.drop(["Id","Species"],axis=1)

#centered around origin
X_std=StandardScaler().fit_transform(X)

#pca=PCA(n_components=4)


##components has been reduced to 2 because we got 95% variance in first two component
##testing with n_components=4

pca=PCA(n_components=2)


X_transform=pca.fit_transform(X_std)

print (pca.explained_variance_ratio_)

#print (X_transform)

#extracting first & second coloumn of the X_transform

pca1=list(zip(*X_transform))[0]
pca2=list(zip(*X_transform))[1]

color_dict={}
color_dict["Iris-setosa"]="green"
color_dict["Iris-versicolor"]="red"
color_dict["Iris-virginica"]="blue"

i=0
for label in labels:
     plt.scatter(pca1[i],pca2[i],color=color_dict[label])
     i=i+1

plt.show()          