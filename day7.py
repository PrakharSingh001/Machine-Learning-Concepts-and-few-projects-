#Decision Tree
import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\salary_more_than_100k.csv')
y=df.salary
x=df.drop(['salary'],axis='columns')
dummies=pd.get_dummies(df.Company,df.job,df.degree)
new=pd.concat([x,dummies],axis='columns')
X=new.drop(['Company','job','degree'],axis='columns')
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
score=model.score(x_train,y_train)
predict=model.predict([[2,0,0]])
print(predict)
print(score)
