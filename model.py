import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("fiat500.csv")
data.shape
data.tail(10)
data.describe()
list(data)
data["previous_owners"].unique()
data.groupby(['model']).count()
data.groupby(['previous_owners']).count()
data['model'].unique()
cor=data.corr()
cor
#cor is alays bw -1 and 1
#correlation metrix 
import seaborn as sns
sns.heatmap(cor,vmax=1,vmin=-1,annot=True,linewidths=.5,cmap='bwr')
data1=data.drop(['lat','ID'],axis=1) #unwanted columns removed
#2-3
data2=data1.drop('lon',axis=1)

data2=pd.get_dummies(data2,dtype=int)

y=data2['price']
X=data2.drop('price',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=0) #0.67 data will be for training.

from sklearn.linear_model import LinearRegression
reg = LinearRegression() #creating object of LinearRegression
reg.fit(X_train,y_train) #training and fitting LR object using training data
ypred=reg.predict(X_test) 

from sklearn.metrics import r2_score
r2_score(y_test,ypred)

from sklearn.metrics import mean_absolute_percentage_error as mape
mape_value = mape(y_test, ypred)
mape_value
from sklearn.metrics import mean_squared_error #calculating MSE
mean_squared_error(y_test,ypred)
#print(t**.5)
# Saving model to disk
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))