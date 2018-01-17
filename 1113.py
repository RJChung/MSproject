#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:09:07 2017

@author: renzhengzhong
"""

#新聞資料處理

import os
import re
import nltk
import numpy as np
from nltk.collocations import *

'''匯入 VIX Data '''
#Log in VIX Data ----------------------------------------------------
import pandas as pd  #Data Frame
# call 值的方式 : VIX.ix['2004/1/2','VIX Close'] --> call 出VIX值
               # VIX.index --> 跑出所有VIX有值的日期
VIX = pd.read_csv('/Users/renzhengzhong/Desktop/tidy_data/vix_test.csv',index_col = 0)

#--------------------------------------------------------------------

'''存取三大新聞資源所屬資料夾下方之所有檔案'''

#取得每個source每年份之檔案名稱
Path = '/Users/renzhengzhong/Desktop/tidy_data'
class Docpath():
    def __init__(self, path, name):
        self.path = (path+'/'+name)
        self.name = name
        
    def getPath(self,Dir):
        try:
            path = [os.path.join(Dir,each) for each in os.listdir(Dir)]
        except NotADirectoryError:
            pass
        return path
     
    def getYear(self,path):
        try:
            yearPath = self.getPath(os.path.join(path, self.name))
        except:
            print('Not a correct dirPath')
        return yearPath
        
    def getDoc(self,yearly_path=[]):
        Doc = []
        for each in yearly_path:
            try:
                Doc.append(self.getPath(each))
            except:
                print('Not a correct dirPath:', each)
                pass
        return Doc
        #del Doc

def Get_News(Dict_):
    #透過路徑開啟資料
    with open(Dict_,'r') as f:
        tmp = f.read()
    
    #年月日表達法
    if 'yyyy'  in tmp:
        #用來處理特殊的日期格式：在2015/201521 以及 2016/201621會遇到
        tmp = tmp.replace("\\f1 yyyy",'yyyy')
        tmp = tmp.replace("\\f1 mmmm",'mmmm')
        tmp = tmp.replace("\\f1 dddd",'dddd')
        tmp = tmp.replace("\\f0  ",'')
        dates = r'[0-9]+ \n[y]+\n[0-9]+ \n[m]+\n[0-9]+ \n[d]+'
        dates= re.findall(dates,tmp)
        dates = [dates[each].replace('\n','') for each in range(len(dates))]
        
    else:
        tmp=tmp.replace("\\f2 \\\'a6\\\'7e",'yyyy')
        tmp=tmp.replace("\\f2 \\\'a4\\\'eb",'mmmm')
        tmp=tmp.replace("\\f2 \\\'a4\\\'e9",'dddd')
        tmp=tmp.replace("\\f1 ",'')
        # Regular Expressio for finding out the days! 抓出100筆資料(有些不滿100筆)
        dates = r'[0-9]+ \n[y]+\n [0-9]+ \n[m]+\n [0-9]+ \n[d]+'
        dates= re.findall(dates,tmp)
        dates = [dates[each].replace('\n','') for each in range(len(dates))]
    
    
    #透過日期來當成dic的key --> ex: (2015012201 : XXXXXXXXXXXXX)
    Doc={}
    for each in dates:
        i = 0 
        if each+'_'+str(i) not in Doc.keys():
            Doc[each+'_'+str(i)] = []
    
        else:
            while each+'_'+str(i) in Doc.keys():
                i = i+1    
            Doc[each+'_'+str(i)] = []
            
    tmp = tmp.replace('\\','')
    
    # 找尋index2的位置
    def find_Endindex(i1,i21,i22,i23):
        index2 = [i21.start(),i22.start(),i23.start()] 
        count = 0
        mid = min(index2)
        for i in index2:
            if max(index2)<i<max(index2):
                mid = i 
            if i>1:
                count = count+1
        if count ==3:
            return min(index2)
        elif count==1:
            return max(index2)
        else:
            return mid
    
    # 實作：找尋每則新聞index的區間為何
    startpoint = 0
    no = 0
    for each in Doc.keys():
        try: #每篇文章開頭(index1)與結尾(index1+index2)的擷取

            i12 = re.search(' Dow Jones & Company, Inc.',tmp[startpoint:])            
            index1 = i12.end()+startpoint

            if re.search('\{field\{\*fldinst',tmp[index1:]) != None:
                i23 = re.search('\{field\{\*fldinst',tmp[index1:])

            index2 = i23.start()+index1
            news = tmp[index1:index2]
            Doc[each] = news
            startpoint = index2
            no = no + 1
        except AttributeError:
            print('AttributeError Happend')
            print("Path.{}".format(Dict_))
            
    #格式清理----------
    for ID,each in Doc.items():
        try:
            each = each.replace(')','')
            each = each.replace(' All Rights Reserved.', '')
            each = each.replace('\n\n','')
            each = each.replace('contributed to this article.','')
            each = each.replace('pardpardeftab360ri0partightenfactor0','')
            each = each.replace('Corrections &Amplifications','')
            each = each.strip()
            each = each.lower()
            
            #regular expression
            filter_regex = re.compile(r'[^a-zA-Z ]') #只留英文字母 其餘皆不留
            each = filter_regex.sub(' ',each)
            Doc[ID] = each
        except AttributeError:
            print('AttributeError Happend')
            
    return Doc

# Convert every news from string to list
def ToCorpus(Doc):
    News={}
    try:
        for eachFile in Doc: #從list --> Dic
            for ID,news in eachFile.items(): #從 Dic -->dic
                eachnews = news.split(' ')
                if len(eachnews)>850:
                    News[ID] = eachnews
                    
                else:
                    pass
    except:
        pass
        print('進入except')
    return News
 #先建立stopwords的清單
with open('/Users/renzhengzhong/Desktop/tidy_data/stopwords.txt','r') as sw:
    Stopwords = sw.readlines()
    Stopwords = [each.strip() for each in Stopwords]

#Filter words that belong to stopwords
def filter_regex(word_dict):
    #regular expression
    filter_regex = re.compile(r'[^a-zA-Z ]') #只留英文字母 其餘皆不留
    news_list=[]
    news_dict={}
    for ID,wordlist in word_dict.items():
        for eachword in wordlist:
            news_list.append(filter_regex.sub(' ',eachword))
        news_dict[ID] = news_list
        news_list = []
    return news_dict

#Filter words that belong to stopwords
def filter_stopwords(word_dict):
    #將屬於停止詞彙的字給篩除
    news_list=[]
    news_dict={}
    for ID,wordlist in word_dict.items():
        for eachword in wordlist:
            if eachword not in Stopwords and len(eachword)>2:
                news_list.append(eachword)
            else:
                pass
        news_dict[ID] = news_list
        news_list = []
    return news_dict

'''
def Bigram(word_dict):
    bigram = nltk.collocations.BigramAssocMeasures()
    #要怎麼去訓練bi-gram ? 思考
    for ID,doc in word_dict.items():
        finder = BigramCollocationFinder.from_words(doc)
        finder.nbest(bigram.pmi,10)
'''
#Word Countings------------------------------------------------

def toCorpus(Doc_Dict):
    corpus = []
    for ID,eachdoc in Doc_Dict.items():
        tmp = str()
        for eachword in eachdoc:
            tmp = tmp+eachword+' '
        corpus.append(tmp)
    return corpus

#--------------------------------------------------------------------
# TF-IDF implement
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  

def tf_idf(corpus):    
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tmp = str()
    CORPUS = []
    for ID,each in corpus.items():
        for wd in each:
            tmp = tmp +" " +  wd
        CORPUS.append(tmp)
        tmp = str()
    corpus = CORPUS
    tf = vectorizer.fit_transform(corpus).toarray()
    tfidf = transformer.fit_transform(tf)
    word = vectorizer.get_feature_names()  #所有文本的關鍵字
    weight = tfidf.toarray() #對應的tfidf矩陣    
    with open('/Users/renzhengzhong/Desktop/tfidf0115.txt','w',encoding = 'utf-8') as f:
        for i in range(len(weight)):
            for j in range(len(word)):
                if j == 0:
                    f.write("doc_{}".format(i) + '\n')                    
                if weight[i][j]>0.2 and tf[i][j]>4: #只取出現過4次以上的  
                    f.write(word[j]+'\n')
    return tfidf,word   # 回傳的tf-idf為一個稀疏矩陣 -->再去查詢關於稀疏矩陣的用法

#--------------------------------------------------------------------

#利用 TFIDF所萃取出來的集合當成是特徵值
def tfidf_feature_matrix(tfidf_doc_path, Bow): #根據目前情況, 我會把bow丟入 Bow中 
    with open(tfidf_doc_path,'r', encoding = 'utf-8') as f:
        total_feature = []
        for line in f:
            if line[0:4] == 'doc_':  #每天的feature會在doc_i 之後開始
                pass
            else:
                total_feature.append(line.strip())
        total_feature = list(set(total_feature))
    #return total_feature        
    #make a matrix table( with only raw = doc and col = feature, all default value =0 )

    #feature_table = np.array(total_feature)
    Feature_Table = pd.DataFrame()
    #fill in the table
    column_feature = []
    for feature in total_feature:
        for ID,each in Bow.items():
            if feature in each:
                column_feature.append(1)
            else:
                column_feature.append(0)
        column_feature = np.array(column_feature).transpose()
        Feature_Table.insert(0,feature,column_feature)
        column_feature = []
        
    return Feature_Table
#--------------------------------------------------------------------

#Log in Sentiment Words----------------------------------------------
def senti_word(word_path): # 預計放入五種情緒詞: pos, neg, uncertain, litigious, modal strong
    with open(word_path,'r') as wd:
        WORD = []
        WD = wd.readlines()
        for each in WD:
            WORD.append(each.lower().strip())
    return WORD

#將五個情緒詞庫匯入
pos = senti_word('/Users/renzhengzhong/Desktop/tidy_data/pos.txt')
neg = senti_word('/Users/renzhengzhong/Desktop/tidy_data/neg.txt')
unc = senti_word('/Users/renzhengzhong/Desktop/tidy_data/unc.txt')
litigious = senti_word('/Users/renzhengzhong/Desktop/tidy_data/litigious.txt')
strong = senti_word('/Users/renzhengzhong/Desktop/tidy_data/strong.txt')
#--------------------------------------------------------------------


#Bag of words----------------------------------------------------

def tidy_ID(ID):
    tmp = re.search(' dddd_',ID)
    index = tmp.start()
    ID = ID[0:index]
    ID = ID.replace(' yyyy ','/')
    ID = ID.replace(' mmmm ','/')
    ID = ID.replace(' yyyy','/')
    ID = ID.replace(' mmmm','/')
    return ID

'''將一天內的所有文章變成一篇大文章: Bag of words 的概念'''
def BOW(dict_doc):
    bow = {}
    tmp=[]
    for ID,eachdoc in dict_doc.items():
        ID = tidy_ID(ID)
        for eachword in eachdoc:
            tmp.append(eachword)
        if  ID in bow.keys() :
            bow[ID] = bow[ID] + tmp
        else:
            bow[ID] = tmp
        tmp = []
    return bow
#--------------------------------------------------------------------
'''
#NVIX by sentiment analysis----------------------------------------------------
def nvix(BOW,VIX):
    pos = senti_word('/Users/renzhengzhong/Desktop/tidy_data/pos.txt')
    neg = senti_word('/Users/renzhengzhong/Desktop/tidy_data/neg.txt')
    unc = senti_word('/Users/renzhengzhong/Desktop/tidy_data/unc.txt')
    litigious = senti_word('/Users/renzhengzhong/Desktop/tidy_data/litigious.txt')
    strong = senti_word('/Users/renzhengzhong/Desktop/tidy_data/strong.txt')
    score = {}
    tmp = 0
    class_count=[0,0,0,0,0]
    count = {}
    wdcount = 0
    for ID,Aggre_Doc in BOW.items():
        for eachword in Aggre_Doc:
            if eachword in neg:
                tmp = tmp - 1
                class_count[0]=class_count[0]+1
            elif eachword in pos:
                tmp = tmp - 0.8 
                class_count[1]=class_count[1]+1                
            elif eachword in unc:
                tmp = tmp - 0.5
                class_count[2]=class_count[2]+1
            elif eachword in litigious:
                tmp = tmp - 1.5
                class_count[3]=class_count[3]+1
            elif eachword in strong:
                tmp = tmp - 2
                class_count[4]=class_count[4]+1
            else:
                wdcount = wdcount-1
            wdcount = wdcount+1
        score[ID] = tmp/wdcount
        class_count = [i/wdcount for i in class_count ]
        count[ID] = class_count
        tmp = 0 
        wdcount = 0
        class_count = [0,0,0,0,0]
    return score,count
'''
#--------------------------------------------------------------------

#Get the proper data dates-------------------------------------------
def get_Date(news_Dict,VIX):    
    Date = []
    for ID,eachday in news_Dict.items():
        if ID in VIX.index:
            Date.append(ID)
    return Date
#--------------------------------------------------------------------

#modified date of VIX------------------------------------------------
def get_modified_BOW(BOW, Date):
    tmp={}
    for each in Date:
        if each in BOW.keys() :
            tmp[each] = BOW[each]
    modified_BOW = tmp
    return modified_BOW
#--------------------------------------------------------------------


#modified date of VIX------------------------------------------------
def get_modified_VIX(VIX, Date):
    tmp={}
    for each in Date:
        if each in VIX.index :
            tmp[each] = VIX.ix[each][0]
    modified_VIX = tmp
    return modified_VIX
#--------------------------------------------------------------------

#Wall Street Journals------------------------------------------------
#WSJ的所有檔案路徑
WSJ = Docpath(Path,'WSJ')
WSJYear = WSJ.getYear(Path)
WSJDoc = WSJ.getDoc(WSJYear)

DOC_WSJ=[]
for eachyear in WSJDoc:
    for eachfile in eachyear:
        try:
            DOC_WSJ.append(Get_News(eachfile))
        except UnicodeDecodeError:
            print('Error File{}'.format(eachfile))
            
        except UnboundLocalError:
            print('Error File{}'.format(eachfile))

DOC_WSJ = ToCorpus(DOC_WSJ)
DOC_WSJ = filter_regex(DOC_WSJ)
DOC_WSJ = filter_stopwords(DOC_WSJ)
DOC_Corpus = toCorpus(DOC_WSJ)
WSJ_BOW = BOW(DOC_WSJ)
Date = get_Date(WSJ_BOW, VIX)
bow = get_modified_BOW(WSJ_BOW,Date)
vix = get_modified_VIX(VIX, Date)    
WSJ_tfidf,feature = tf_idf(bow)
Table = tfidf_feature_matrix('/Users/renzhengzhong/Desktop/tfidf0115.txt', bow) 
Table.insert(0,'VIX',vix.values()) #把vix(3169)筆加入dataframe中
# NVIX,N_count = nvix(bow,vix)   #先連同上面的 nvixfunction一起註解掉

#--------------------------------------------------------------------
#try to learn...
#FEATURES_DF = pd.DataFrame(columns = ['VIX','Neg','Pos','Unc','Ligitious','Strong'])

Date = []
Neg = []
Pos = []
Unc = [] 
Litigious = []
Strong = []

for ID,day in N_count.items():
    Date.append(ID)
    Neg.append(day[0])
    Pos.append(day[1])
    Unc.append(day[2])
    Litigious.append(day[3])
    Strong.append(day[4])
Feat_dict = {'Date' : Date,
             'Neg':Neg,
             'Pos':Pos,
             'Unc':Unc,
             'Litigious':Litigious,
             'Strong':Strong}
pseudo_vix = VIX['VIX Close'][0:3169]

Features_DF = pd.DataFrame(Feat_dict)
Features_DF['VIX'] = pseudo_vix.values

'''做出上漲下跌的 0 1 數值'''
from pandas import Series
Up_Down=[]
for i in range(len(Features_DF)):
    if i>=1 and i <= len(Features_DF):
        if Features_DF['VIX'][i] > Features_DF['VIX'][i-1]:
            Up_Down.append(1)
        else:
            Up_Down.append(0)
    else:
        Up_Down.append(0)
VIX_Up_Down = Series(Up_Down, name = 'VIX_Up_Down')
Features_DF = pd.concat([Features_DF, VIX_Up_Down],axis = 1)
del Up_Down
#__________________________________________________________________

'''訓練與測試資料的準備'''
from sklearn import cross_validation, metrics   
# 建立訓練與測試資料
train_X, test_X, train_y, test_y = \
'''
cross_validation.train_test_split(Features_DF.iloc[:,1:5],\
                                  Features_DF.iloc[:,7],\
                                  test_size = 0.3) #choose the testing data size 
'''                               
train_X, test_X, train_y, test_y = \
cross_validation.train_test_split(Table.iloc[:,1:-1],\
                                  Table.iloc[:,0],\
                                  test_size = 0.2) #choose the testing data size 
                                  
'''隨機森林分類器'''
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# 建立 random forest 分類器
forest = RandomForestClassifier(n_estimators = 4) #n_jobs=-1,max_features='auto', \
                                #n_estimators = 3,random_state = 0)
# 訓練random forest分類器
forest_fit = forest.fit(train_X, train_y)

# 預測
RanFor_test_y_predicted = forest.predict(test_X)
# 績效
RF_self = forest.predict(train_X)
cm_self = metrics.confusion_matrix(train_y, RF_self)
RanFor_accuracy = metrics.accuracy_score(test_y, RanFor_test_y_predicted)
cm = metrics.confusion_matrix(test_y, RanFor_test_y_predicted)
print('Accuracy for Random Forests  = ',RanFor_accuracy)  
print('TRAIN SCORE: ',forest.score(train_X, train_y),' TEST SCORE: ', forest.score(test_X, test_y))


#__________________________________________________________________
''' SVM 分類器 '''
from sklearn import svm
# 建立向量支持器 分類器
# SVC參數kernel:它指定要在算法中使用的內核類型,
# 有:'linear','poly','rbf'(default),'sigmoid','precomputed'
svc = svm.SVC(kernel='rbf', C= 0.5)
svc_fit = svc.fit(train_X, train_y)

# 預測
svc_test_y_predicted = svc.predict(test_X)
# 績效
svc_accuracy = metrics.accuracy_score(test_y, svc_test_y_predicted)
print('Accuracy for SVM = ',svc_accuracy) 
#使用kernel='linear', 再加入辭典(.strip())
print('TRAIN SCORE: ',svc.score(train_X, train_y),' TEST SCORE: ', svc.score(test_X, test_y))
#__________________________________________________________________

'''隨機森林回歸''' #(用5個特徵為度 顯示為非常不準確, 完全沒有預測力)
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
RFC_X, RFC_Y = make_regression(n_features = 100)
RFC = RandomForestRegressor(max_depth = 20)
RFC.fit(train_X, train_y)

print(RFC.feature_importances_)
RFC_train = RFC.predict(train_X)
RFC_test = RFC.predict(test_X)
train_score = r2_score(train_y, RFC_train)
test_score = r2_score(test_y, RFC_test)

#__________________________________________________________________

'''SVR regression''' #(用5個特徵為度 顯示為非常不準確, 完全沒有預測力)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# 建立支持向量機器SVR
# SVR參數kernel:它指定要在算法中使用的內核類型,
# 有:'linear','poly','rbf'(default),'sigmoid','precomputed'
svr = SVR()
svr_fit = svr.fit(train_X, train_y)

# 預測
svr_test_y_predicted = svr.predict(test_X)
# 績效
PCCs = np.corrcoef(svr_test_y_predicted, test_y)
RMSE = (mean_squared_error(test_y,svr_test_y_predicted))**(1/2)
R_squared = r2_score(test_y,svr_test_y_predicted)
print(R_squared)
print(PCCs)
print(RMSE)
#__________________________________________________________________









#New York Times------------------------------------------------------
#NYT的所有檔案路徑
NYT = Docpath(Path,'NYT')
NYTYear = NYT.getYear(Path)
NYTDoc = NYT.getDoc(NYTYear)

DOC_NYT=[]
for eachyear in NYTDoc:
    for eachfile in eachyear:
        try:
            DOC_NYT.append(Get_News(eachfile))
        except UnicodeDecodeError:
            print('Error File{}'.format(eachfile))
            
        except UnboundLocalError:
            print('Error File{}'.format(eachfile))
            
#--------------------------------------------------------------------



#Financial Times-----------------------------------------------------
#FT的所有檔案路徑
FT = Docpath(Path,'FT')
FTYear = FT.getYear(Path)
FTDoc = FT.getDoc(FTYear)

DOC_FT=[]
for eachyear in FTDoc:
    for eachfile in eachyear:
        try:
            DOC_FT.append(Get_News(eachfile))
        except UnicodeDecodeError:
            print('Error File{}'.format(eachfile))
            
        except UnboundLocalError:
            print('Error File{}'.format(eachfile))
