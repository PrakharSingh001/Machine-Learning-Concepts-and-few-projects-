#Saving Predicted Model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import  linear_model
import pickle
#from sklearn.externals import  joblib
#df=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\Book1.csv')
#print(df)
#plt.xlabel('area')
#plt.ylabel('Price')
#plt.title('House Price A/Q Area')
#plt.scatter(df.area,df.price)
#plt.show()
'''reg=linear_model.LinearRegression()
reg.fit(df[['area']].values,df.price)
print(reg.predict([[5000]]))'''
'''with open('model_pickel','wb') as f:
    pickle.dump(reg,f)'''
with open('model_pickel','rb') as f:
    mp=pickle.load(f)
print(mp.predict([[900]]))
#joblib for larger arrays
#joblib.dump(model,'prakhar sing')

