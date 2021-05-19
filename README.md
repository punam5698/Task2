#Importing required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets
from sklearn.cluster import KMeans

#To ignore the warning
import warnings as wg
wg.filterwarnings("ignore")



#Reading dataset

data=pd.read_csv("Iris.csv")
data.head()
data.info
data.shape
corr=data.corr()
data.describe()


x=data.iloc[:,:-1].values
y=data.iloc[:,:-1].values
print("X")
print(x)
print("Y")
print(y)



#Visuallizing the dataset

sns.heatmap(data.corr())

sns.pairplot(data)

x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
sns.scatterplot(x,y,color='orange')
plt.title("SepalLengthCm vs SepalWidthCm")

x=data["PetalLengthCm"]
y=data["PetalWidthCm"]
sns.scatterplot(x,y,color='purple')
plt.title("PetalLengthCm vs PetalWidthCm")


#Finding optimum number of clusters for K-Means classification

x=data.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=5)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("wcss")

plt.show()



#Applying K-Means to the dataset

kmeans=KMeans(n_clusters=3,init='k-means++',
             max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)



#Visuallizing the clusters

plt.figure(figsize=(10,8))
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=30,c='red',label="Iris-seatosa")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=30,c='blue',label="Iris-versicolour")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=30,c='green',label="Iris-verginica")

plt.legend()
plt.show()


#RESULT : Thus the optimum number of clusters for the Iris dataset is 3.
