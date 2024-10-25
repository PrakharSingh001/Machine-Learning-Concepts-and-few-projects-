from sklearn import linear_model
import numpy as np
x=[[97],[98],[99],[100]]
y=[2010,2011,2012,2013]
model=linear_model.LinearRegression()
model.fit(x,y)
predicted=model.predict([[2010]])
print(predicted)
