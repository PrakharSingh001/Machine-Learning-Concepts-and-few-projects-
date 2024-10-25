import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import  linear_model
df=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\Book1.csv')
print(df)
#plt.xlabel('area')
#plt.ylabel('Price')
#plt.title('House Price A/Q Area')
#plt.scatter(df.area,df.price)
#plt.show()
x=np.array([df['area']])
reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.predict([[4000]]))
