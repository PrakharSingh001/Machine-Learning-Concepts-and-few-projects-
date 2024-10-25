from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits=load_digits()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.1,random_state=10)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(pred)
#confusion matrix help to know where your model is wrong
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,pred)
print(cm)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
