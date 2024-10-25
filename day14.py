from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
#BuildingArea     
#Bedroom2         
#Bathroom          
data=pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\Melbourne_housing_FULL.csv')
print(data.head())
print(data.nunique())
coll_to_use=['Suburb','Rooms','Price','Distance','Bedroom2','Bathroom','Car','Landsize','BuildingArea','Propertycount']
dataset=data[coll_to_use]
print(dataset)
print(dataset.isna().sum())
fill_zero=['Car','Distance','Propertycount','Bedroom2','Bathroom']
dataset[fill_zero]=dataset[fill_zero].fillna('0')
print(dataset.head())

dataset['Landsize']=dataset['Landsize'].fillna(dataset['Landsize'].mean())
print(dataset.isna().sum())

dataset['BuildingArea']=dataset['BuildingArea'].fillna(dataset['BuildingArea'].mean())
print('             ')

dataset=dataset.dropna()
dummies=pd.get_dummies(dataset)
concat=pd.concat([dataset,dummies],axis='columns')
x=concat.drop(['Price','Suburb'],axis='columns')
y=concat.Price
print(dataset.isna().sum())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20)
lass=Lasso()
lass.fit(x_train,y_train)
print(lass.score(x_test,y_test))

rid=Ridge()
rid.fit(x_train,y_train)
print(rid.score(x_test,y_test))
