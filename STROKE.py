import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('stroke_data.csv')
print(data.info())
print(data.columns)
col=data.columns.values
data['sex']=data.sex.fillna(data.sex.median())
print(data.isna().sum())
for i in col:
    sn.boxplot(data[i])
    plt.show()
    if len(data[i].value_counts().values) <= 5:
        sn.countplot(data[i])
        plt.show()
#hypertension 1
#heart disease 1
#ever married 1
#work type 2
#bmi
data['z-score']=(data.bmi-data.bmi.mean())/data.bmi.std()
df=data[(data['z-score'] > -3)&(data['z-score'] <3)]
print(df.shape)
print(data.shape)
q1=df.bmi.quantile(0.25)
q3=df.bmi.quantile(0.75)
iqr=q3-q1
upper=q3+1.5*iqr
lower=q1-1.5*iqr
df=df[(df.bmi < upper) &(df.bmi >lower)]
qa1=df.bmi.quantile(0.25)
qa3=df.bmi.quantile(0.75)
Iqr=qa3-qa1
u=qa3+1.5*Iqr
l=qa1-1.5*Iqr
df=df[(df.bmi < u)&(df.bmi >l )]
print(df.shape)
sn.boxplot(df.bmi)
plt.show()
colm=df.columns.values
x=df[['sex','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']]
y=df['stroke']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)
log_predict=lg.predict(x_test)
from sklearn.tree import DecisionTreeClassifier
dtree_classi=DecisionTreeClassifier()
dtree_classi.fit(x_train,y_train)
dtree_predict=dtree_classi.predict(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn_classi=KNeighborsClassifier()
knn_classi.fit(x_train,y_train)
knn_pred=knn_classi.predict(x_test)
print(lg.score(x_test,y_test))
print(knn_classi.score(x_test,y_test))
print(dtree_classi.score(x_test,y_test))
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(log_predict,marker='o',color='blue',label='logistic_reg_prediction')
plt.title('Logistic regression prediction vs Y_test')
plt.legend()
plt.show()
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(knn_pred,marker='o',color='blue',label='KNN_Classification_prediction')
plt.title('KNN Classification prediction vs Y_test')
plt.legend()
plt.show()
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(dtree_predict,marker='o',color='blue',label='decision_tree_prediction')
plt.title('Decision tree classification prediction vs Y_test')
plt.legend()
plt.show()
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x_train.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=1,activation=keras.activations.sigmoid))
models.compile(optimizer='adam',metrics='accuracy',loss=keras.losses.binary_crossentropy)
hist=models.fit(x_train,y_train,batch_size=20,epochs=200,validation_split=0.45)
plt.plot(hist.history['accuracy'],label='training accuracy',marker='o',color='red')
plt.plot(hist.history['val_accuracy'],label='val_accuracy',marker='o',color='blue')
plt.title('Training Vs  Validation accuracy')
plt.legend()
plt.show()