import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\k_means.csv')

scaler=MinMaxScaler()
scaler.fit(df[['Income']])
df.Income=scaler.transform(df[['Income']])

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
print(df.head())

km=KMeans(n_clusters=3)
y=km.fit_predict(df[['Age','Income']])
print(y)

df['cluster']=y
print(df.head())
print(km.cluster_centers_)
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1.Income,color='green')
plt.scatter(df2.Age,df2.Income,color='red')
plt.scatter(df3.Age,df3.Income,color='black')
plt.show()
