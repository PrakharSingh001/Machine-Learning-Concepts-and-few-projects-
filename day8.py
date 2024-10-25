import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
#print(iris.target )
print('features=',iris.feature_names)
df=pd.DataFrame(iris.data,columns=iris.feature_names )
#print(df.head())
df['target']=iris.target 
print(df.head())
print('target names=',iris.target_names)
#print(df[df.target==1].head)
df['flower_name']=df.target.apply(lambda x: iris.target_names[x] )
print(df.head())
#SVM is used for classification purpose, to classify between
#few objects using data frames
from sklearn.model_selection import train_test_split
X=df.drop(['target','flower_name'],axis='columns')
Y=df.target
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=20)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
