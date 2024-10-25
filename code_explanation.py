import pandas as pd

data=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\code_explain.csv',encoding='ISO-8859-1')
data=data.dropna()
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
X=data['code']
Y=data.drop(['code'],axis='columns')
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
vect=Pipeline([('vectorizer',TfidfVectorizer()),('nb',LogisticRegression())])
vect.fit(x_train,y_train)
z=vect.predict(["print('hi')"])
print(vect.score(x_test,y_test))
print(z)
import pickle
    
    