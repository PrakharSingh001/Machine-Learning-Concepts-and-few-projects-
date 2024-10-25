from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits=load_digits()
'''x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.3,random_state=20)
lr=LogisticRegression()
lr.fit(x_train,y_train)
print('logistic',lr.score(x_test,y_test))

linr=LinearRegression()
linr.fit(x_train,y_train)
print('linear',linr.score(x_test,y_test))

sv=SVC()
sv.fit(x_train,y_train)
print('svc',sv.score(x_test,y_test))

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print('decision',dt.score(x_test,y_test))

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print('random',rf.score(x_test,y_test))
'''
#train_test_split is not much better in model selection because data is continuosuly changing resulting in different scores
#from sklearn.model_selection import KFold
#kf=KFold(n_splits=3)

'''def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)
from sklearn.model_selection import StratifiedKFold
folds=StratifiedKFold(n_splits=10)
scores_l=[]
scores_svm=[]
scores_rf=[]

for  train_index,test_index in folds.split(digits.data,digits.target):
    x_train,x_test,y_train,y_test=digits.data[train_index],digits.data[test_index], \
                                   digits.target[train_index],digits.target[test_index]
    scores_l.append( get_score(LogisticRegression(),x_train,x_test,y_train,y_test))
    scores_svm.append(get_score(SVC(),x_train,x_test,y_train,y_test))
    scores_rf.append( get_score(RandomForestClassifier(),x_train,x_test,y_train,y_test))
print('logistic regression')
print(scores_l)
print('svm')
print(scores_svm)
print('random forest')
print(scores_rf)'''
from sklearn.model_selection import cross_val_score
x=cross_val_score(LogisticRegression(),digits.data,digits.target)
y=cross_val_score(SVC(),digits.data,digits.target )
z=cross_val_score(RandomForestClassifier(),digits.data,digits.target )
print('logisitc',x)
print(x.min())
print(x.max())
print('svm',y)
print(y.min())
print(y.max())
print('random forest',z)
print(z.min())
print(z.max())
