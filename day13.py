import pandas as pd
data=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\spam.csv')
def fun(x):
    if x=='spam':
        return 1
    else:
        return 0
data['spam']=data['Category'].apply(fun)
print(data.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.Message,data.spam,test_size=0.2,random_state=20)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
clf=Pipeline([
    ('Vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
    ])
g=cross_val_score(clf,data.Message,data.spam)
print(g)
clf.fit(x_train,y_train)
z=['20% discount on clothes']
pred=clf.predict(z)
print(pred)
if pred==0:
    print('not spam')
if pred==1:
    print('spam')
print(clf.score(x_test,y_test))

