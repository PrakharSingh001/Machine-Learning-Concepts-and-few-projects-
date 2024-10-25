import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()
print(dir(digits))
df=pd.DataFrame(digits,columns=digits.feature_names)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2,random_state=10)
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(model.score(x_test,y_test))
