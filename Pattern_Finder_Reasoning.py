from sklearn.linear_model import LinearRegression
import pandas as pd
read=pd.read_csv('pattern.csv')
x=read.Pattern
y=read.Result
model=LinearRegression()
model.fit(x,y)
predict=model.predict([[8+9]])
print(predict)
