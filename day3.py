#Text in number data
import pandas as pd
from sklearn.linear_model import LinearRegression
read=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\dummy_variable.csv')
dummies=pd.get_dummies(read.town)
merged=pd.concat([read,dummies],axis='columns')
print(merged)
df=merged.drop(['town','west windsor'],axis='columns')

#Area prediction of a house of a particular place
model=LinearRegression()
X=df.drop(['area'],axis='columns')
print(X)
Y=df.area
model.fit(X,Y)
dic=model.predict([[590775.63964739,0,1]])
print(dic)

print(model.score(X,Y))

#price prediction
x=df.drop(['price'],axis='columns')
y=df.price

model.fit(x,y)
pred=model.predict([[3400,0,0]])
print(pred)
print(model.score(x,y))

