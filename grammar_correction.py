import pandas as pd
import spacy
nlp=spacy.load('en_core_web_lg')
data=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\Grammar_check.csv',encoding='ISO-8859-1')
X = data['Wrong']
print(data['Error'].value_counts())
def div(x):
    if x=='Incorrect Tense Usage':
        return 0
    if x=='Preposition Error':
        return 1
    if x=='Adverb Error':
        return 2
    if x=='Correct':
        return 3
    if x=='Subject-Verb Agreement':
        return 4
data['Errors']=data['Error'].apply(div)
Y=data['Errors']
print(Y.value_counts())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=20)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import  DecisionTreeClassifier

pipe = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier',DecisionTreeClassifier())])

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print(pipe.score(x_test, y_test)) 
text='He began walking again, eager to explore, but soon realized he had no idea where he was'
predicted=pipe.predict([text])
import pickle
with open('grammar_correction','wb') as file:
    pickle.dump(pipe,file)
if predicted==0:
    print('Incorrect Tense Usage ')
if predicted==1:
    print('Preposition Error  ')
if predicted==2:
    print(' Adverb Error ')
if predicted==3:
    print(' Correct ')
if predicted==4:
    print(' Subject-Verb Agreement ')
    
          
