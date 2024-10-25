import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\titanic.csv')
data=data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis='columns')
dummies=pd.get_dummies(data.Sex)
data=pd.concat([data,dummies],axis='columns')
print(data.head(3))
X=data.drop(['Survived','Sex'],axis='columns')
Y=data.Survived
print(X.head(3))
print(Y.head(3))
print('got na values in',X.columns[X.isna().any()])
#got na values in Age Column
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
print('printing x_train')
print(x_train)
x_train=x_train.fillna(x_train.mean())
x_test=x_test.fillna(x_test.mean())

y_train=y_train.fillna(y_train.mean())
y_test=y_test.fillna(y_test.mean())
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
p=model.predict(x_test[:10])
print(x_test[:10])
print(y_test[:10])
print(model.score(x_test,y_test))
print(model.predict_proba(x_test[:25]))
