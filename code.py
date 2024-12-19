import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import ensemble, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
data = pd.read_csv('employee_leave_train.csv').drop('Unnamed: 0', axis = 1)
#предобработка
data['Отстранения'] = data['Отстранения'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Пол'] = data['Пол'].apply(lambda x: 1 if x == 'Male' else 0)
categ_columns = [x for x in data.columns if data[x].dtype == 'object']
categ_columns
data = pd.get_dummies(data, categ_columns)
data.info()
y = data['Увольнение'] #признак
x = data.drop('Увольнение', axis = 1)#переменная
#модель
linreq = linear_model.LinearRegression()
rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=7)
gf = ensemble.GradientBoostingRegressor(n_estimators=70, max_depth=5, min_samples_split=10)
linreq.fit(x,y)
rf.fit(x,y)
gf.fit(x,y)
#тестовая часть
data_test = pd.read_csv('employee_leave_test.csv').drop('Unnamed: 0', axis = 1)
data_test['Отстранения'] = data_test['Отстранения'].apply(lambda x: 1 if x == 'Yes' else 0)
data_test['Пол'] = data_test['Пол'].apply(lambda x: 1 if x == 'Male' else 0)
categ_columns1 = [x for x in data_test.columns if data_test[x].dtype == 'object'] 
categ_columns1
data_test = pd.get_dummies(data_test, categ_columns1)
data_test.shape, data.shape
y_test_linreq = linreq.predict(data_test)
y_test_rf = rf.predict(data_test)
y_test_gf = gf.predict(data_test)
y_data_true = np.array(pd.read_csv('sample_submission.csv')).flatten()
plt.hist(y_data_true)
plt.hist(y_test_rf)
