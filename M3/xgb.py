#导入数据
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics,svm,ensemble,linear_model
import sklearn as skl
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from math import isnan
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pylab as plot
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import copy

dataset=pd.read_csv('../selected/train.csv')
test=pd.read_csv('../selected/test.csv')


# 按照训练集正负样本比例抽样选择数据
tmp1 = dataset[dataset.TARGET==1]
tmp0 = dataset[dataset.TARGET==0]
x_valid_1 = tmp1.sample(frac=0.3, random_state=70, axis=0)
x_train_1 = tmp1.drop(x_valid_1.index.tolist())
x_valid_2 = tmp0.sample(frac=0.1, random_state=70, axis=0)
x_train_2 = tmp0.drop(x_valid_2.index.tolist())

x = pd.concat([x_train_1,x_train_2],axis=0)
y = x.pop('TARGET')
x=x.drop(['EID'],axis=1)

val_x = pd.concat([x_valid_1,x_valid_2],axis=0)
val_y = val_x.pop('TARGET')
val_x=val_x.drop(['EID'],axis=1)

test_x=test.drop(['EID'],axis=1)

dval = xgb.DMatrix(val_x,label=val_y)
dtrain = xgb.DMatrix(x, label=y)
dtest = xgb.DMatrix(test_x)

random_seed = 100
params={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'early_stopping_rounds':100,
        'scale_pos_weight': 29092.0/123914.0,
        'eval_metric': 'auc',
        'gamma':0.3,
        'max_depth':5,
        'lambda':50,
        'alpha':1,
        'subsample':0.9,
        'colsample_bytree':0.7,
        'min_child_weight':5, 
        'eta': 0.008,
        'seed':random_seed,
        'nthread':4,
}
    
num_boost_round = 30000

watchlist = [(dtrain, 'train'), (dval,'val')]
bst = xgb.train(params, dtrain, num_boost_round,evals=watchlist,early_stopping_rounds=500)
preds_xgb_test=bst.predict(dtest,ntree_limit=bst.best_ntree_limit)

bst.save_model('xgb_finial.model')

qwe=copy.copy(preds_xgb_test)
qwe[qwe>0.3]=1
qwe[qwe<0.3]=0

test_EID=test['EID'].tolist()
new=pd.DataFrame({'EID':test_EID,
                 'FORTARGET':qwe,
                 'PROB':preds_xgb_test})
new['FORTARGET']=new['FORTARGET'].astype('int')
new.to_csv('xgb_finial.csv',index=False)

feature_importance=pd.Series(bst.get_fscore()).sort_values(ascending=False)
feature_importance.plot(kind='bar', title='Feature Importances')
plt.show()
