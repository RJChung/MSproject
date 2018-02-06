#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:00:23 2018

@author: renzhengzhong
"""

'''訓練與測試資料的準備'''
from sklearn import cross_validation, metrics   
# 建立訓練與測試資料
'''
train_X, test_X, train_y, test_y = \
cross_validation.train_test_split(Features_DF.iloc[:,1:5],\
                                  Features_DF.iloc[:,7],\
                                  test_size = 0.3) #choose the testing data size 
'''                               
train_X, test_X, train_y, test_y = \
cross_validation.train_test_split(Table.iloc[:,2:-1],\
                                  Table.iloc[:,0],\
                                  test_size = 0.2) #choose the testing data size 


#分類器資料
Ctrain_X, Ctest_X, Ctrain_y, Ctest_y = \
cross_validation.train_test_split(cate_table.iloc[:,5:-1],\
                                  cate_table.iloc[:,1],\
                                  test_size = 0.2) #choose the testing data size 

#回歸資料
Rtrain_X, Rtest_X, Rtrain_y, Rtest_y = \
cross_validation.train_test_split(cate_table.iloc[:,5:-1],\
                                  cate_table.iloc[:,4],\
                                  test_size = 0.3) #choose the testing data size 
                                  

Ctrain_X = cate_table.iloc[0:2535,5:-1]
Ctest_X = cate_table.iloc[2535:3169,5:-1]
Ctrain_y = cate_table.iloc[0:2535:,1]
Ctest_y = cate_table.iloc[2535:3169,1]

                                  
'''隨機森林分類器'''
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# 建立 random forest 分類器
forest = RandomForestClassifier()#n_estimators = 500) #n_jobs=-1,max_features='auto', \
                                #n_estimators = 3,random_state = 0)
# 訓練random forest分類器
forest_fit = forest.fit(Ctrain_X, Ctrain_y)

# 預測
rf_predicted = forest.predict(Ctest_X)
# 績效
rf_self = forest.predict(Ctrain_X)
cm_self = metrics.confusion_matrix(Ctrain_y, rf_self)
fr_accuracy = metrics.accuracy_score(Ctest_y, rf_predicted)
rf_cm = metrics.confusion_matrix(Ctest_y, rf_predicted)#, labels = [1,0,-1])
rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(Ctest_y, rf_predicted)
rf_auc = metrics.auc(rf_fpr, rf_tpr)
print('Confusion Matrix:\n{}'.format(rf_cm))
print('auc: {}'.format(rf_auc))
print('TRAIN SCORE: ',forest.score(Ctrain_X, Ctrain_y),' TEST SCORE: ', forest.score(Ctest_X, Ctest_y))
print('#_____________________________________________________________','\n')


#__________________________________________________________________
''' SVM 分類器 '''
from sklearn import svm

# 建立向量支持器 分類器
# SVC參數kernel:它指定要在算法中使用的內核類型,
# 有:'linear','poly','rbf'(default),'sigmoid','precomputed'
svc = svm.SVC(kernel = 'linear')
svc_fit = svc.fit(Ctrain_X, Ctrain_y)

# 預測
svc_predicted = svc.predict(Ctest_X)
# 績效
svc_accuracy = metrics.accuracy_score(Ctest_y, svc_predicted)
svc_cm = metrics.confusion_matrix(Ctest_y, svc_predicted)
svc_cm_train = metrics.confusion_matrix(Ctrain_y, svc.predict(Ctrain_X))
svc_fpr, svc_tpr, svc_thresholds = metrics.roc_curve(Ctest_y, svc_predicted)
svc_auc = metrics.auc(svc_fpr, svc_tpr)
print('TRAIN SCORE: ',svc.score(Ctrain_X, Ctrain_y),' TEST SCORE: ', svc.score(Ctest_X, Ctest_y))
print('svc_cm{}'.format(svc_cm))
print('SVC_auc{}'.format(svc_auc))
print('#_____________________________________________________________','\n')
#__________________________________________________________________

#ADAMBOOST CLASSIFIER
'''ADAMBOOST 分類器'''
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
# Create and fit an AdaBoosted decision tree  # 

abt = AdaBoostClassifier(n_estimators=1200,algorithm="SAMME.R")#DecisionTreeClassifier(max_depth=5),

abt.fit(Ctrain_X, Ctrain_y)

abt_predicted = abt.predict(Ctest_X)
abt_accuracy = metrics.accuracy_score(Ctest_y, abt_predicted)
abt_cm = metrics.confusion_matrix(Ctest_y, abt_predicted)
abt_fpr, abt_tpr, abt_thresholds = metrics.roc_curve(Ctest_y, abt_predicted)
abt_auc = metrics.auc(abt_fpr, abt_tpr)

print('\n','abt_cm{}'.format(abt_cm),'\n')
print('TRAIN SCORE: ',abt.score(Ctrain_X, Ctrain_y),' TEST SCORE: ', abt.score(Ctest_X, Ctest_y),'\n')
print('ABT_auc: {}'.format(abt_auc))
print('#_____________________________________________________','\n')

#__________________________________________________________________
#naive bayes
from sklearn.naive_bayes import GaussianNB,BernoulliNB
gnb = GaussianNB()
gnb = BernoulliNB()
gnb_fit = gnb.fit(Ctrain_X, Ctrain_y)
gnb_predicted = gnb.predict(Ctest_X)
gnb_accuracy = metrics.accuracy_score(Ctest_y, gnb_predicted)
gnb_cm = metrics.confusion_matrix(Ctest_y, gnb_predicted)
gnb_fpr, gnb_tpr, gnb_thresholds = metrics.roc_curve(Ctest_y, gnb_predicted)
gnb_auc = metrics.auc(gnb_fpr, gnb_tpr)

print('\n','gnb_cm{}'.format(gnb_cm),'\n')
print('TRAIN SCORE: ',gnb.score(Ctrain_X, Ctrain_y),' TEST SCORE: ', gnb.score(Ctest_X, Ctest_y),'\n')
print('gnb_auc: {}'.format(gnb_auc))
print('#_____________________________________________________','\n')

#__________________________________________________________________

#回歸資料
Rtrain_X = cate_table.iloc[0:2535,5:-1] #選擇'5' --> up hold down ; 選擇 4: up down
Rtest_X = cate_table.iloc[2335:3169,5:-1]
Rtrain_y = cate_table.iloc[0:2535:,4]
Rtest_y = cate_table.iloc[2335:3169,4]


'''隨機森林回歸''' 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
RFC_X, RFC_Y = make_regression()
RFC = RandomForestRegressor()
RFC.fit(Rtrain_X, Rtrain_y)

# predict
RFC_train = RFC.predict(Rtrain_X)
RFC_test = RFC.predict(Rtest_X)

# performance
train_score = r2_score(Rtrain_y, RFC_train) # train's score
test_score = r2_score(Rtest_y, RFC_test)
pccs = np.corrcoef(RFC_train, Rtrain_y) # train's pccs
PCCs = np.corrcoef(RFC_test, Rtest_y)
rmse = (mean_squared_error(Rtrain_y,RFC_train))**(1/2) # train's RMSE
RMSE = (mean_squared_error(Rtest_y,RFC_test))**(1/2) 
print('train R2: ', train_score)
print('test R2: ', test_score) # paper有
print('train PCCs: ',pccs)
print('test PCCs: ',PCCs) #correlation paper有
print('train RMSE: ',rmse)
print('test RMSE: ', RMSE) # paper 有

#__________________________________________________________________

'''SVR regression''' 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# 建立支持向量機器SVR
# SVR參數kernel:它指定要在算法中使用的內核類型,

svr = SVR(kernel = 'rbf')
svr_fit = svr.fit(Rtrain_X, Rtrain_y)

# 預測
#svr = svr.predict(train_X)
svr_test_y_predicted = svr_fit.predict(Rtest_X)
# 績效
Train_r2 = r2_score(Rtrain_y, svr_fit.predict(Rtrain_X))
print('Train R2: ',Train_r2)
PCCs = np.corrcoef(svr_test_y_predicted, Rtest_y)
RMSE = (mean_squared_error(Rtest_y,svr_test_y_predicted))**(1/2)
R_squared = r2_score(Rtest_y,svr_test_y_predicted)
print('r2:',R_squared)
print(PCCs)
print(RMSE)
#__________________________________________________________________

'''Adaboost Regression'''
from sklearn.ensemble import AdaBoostRegressor
abtR = AdaBoostRegressor()#n_estimators=1000)

abtR.fit(Rtrain_X, Rtrain_y)

abtR_predicted = abtR.predict(Rtest_X)
abtR_PCCs = np.corrcoef(abtR_predicted, Rtest_y)
abtR_RMSE = (mean_squared_error(Rtest_y,abtR_predicted))**(1/2)
abtR_R_squared = r2_score(Rtest_y,abtR_predicted)
print('TRAIN SCORE: ',abtR.score(Rtrain_X, Rtrain_y),' TEST SCORE: ', abtR.score(Rtest_X, Rtest_y),'\n')
print('#_____________________________________________________','\n')

#__________________________________________________________________