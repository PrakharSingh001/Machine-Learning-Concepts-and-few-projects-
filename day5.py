#Logistic Regression- Classification
import pandas as pd
df=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\insurance.csv')
from sklearn.model_selection import train_test_split
x=df[['age']]
y=df[['bought_insurance']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print(y_test)
prediction=model.predict([[9]])
print(prediction)
#check probability
probability=model.predict_proba([[45]])
print(probability)
