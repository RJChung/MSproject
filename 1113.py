#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:09:07 2017

@author: renzhengzhong
"""

#新聞資料處理

import os
import re
import numpy as np

'''匯入 VIX Data '''
#Log in VIX Data ----------------------------------------------------
import pandas as pd  #Data Frame
# call 值的方式 : VIX.ix['2004/1/2','VIX Close'] --> call 出VIX值
               # VIX.index --> 跑出所有VIX有值的日期
#VIX = pd.read_csv('/Users/renzhengzhong/Desktop/tidy_data/vix_test.csv',index_col = 0)
VIX = pd.read_csv('/Users/renzhengzhong/Desktop/vix_test.csv',index_col = 0)
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
#--------------------------------------------------------------------

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
#--------------------------------------------------------------------
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
#--------------------------------------------------------------------

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
#--------------------------------------------------------------------

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
#--------------------------------------------------------------------

# word lemmatization 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


def wd_stem_lemma(category_list):
    wordnet_lemmatizer = WordNetLemmatizer()    
    porter_stemmer = PorterStemmer()
    stem = []
    lemm = []
    for each in category_list:
        lemm.append(wordnet_lemmatizer.lemmatize(each))
        stem.append(porter_stemmer.stem(each))    
    stem = list(set(stem))
    lemm = list(set(lemm))
    return stem, lemm
#--------------------------------------------------------------------

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

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

'''將一天內的所有文章變成一篇大文章: Bag of words 的概念'''
def BOW(dict_doc):
    porter_stemmer = PorterStemmer()
    bow = {}
    tmp=[]
    for ID,eachdoc in dict_doc.items():
        ID = tidy_ID(ID)
        for eachword in eachdoc:
            tmp.append(porter_stemmer.stem(eachword))
        if  ID in bow.keys() :
            bow[ID] = bow[ID] + tmp
        else:
            bow[ID] = tmp
        tmp = []
    return bow
#--------------------------------------------------------------------
def word_category(pos, neg, unc, strong, litigious):
    five_category = [pos, neg, unc, strong, litigious]
    category_feature = []
    for category in five_category:
        for each in category:
            category_feature.append(each)
    return category_feature
#--------------------------------------------------------------------
#利用 category當成特徵值
def category_feature_matrix(category_feature, Bow): #根據目前情況, 我會把bow丟入 Bow中 
    #feature_table = np.array(total_feature)
    category_feature = list(set(category_feature))
    Feature_Table = pd.DataFrame()
    #fill in the table
    column_feature = []
    for feature in category_feature:
        for ID,each in Bow.items():
            if feature in each:
                column_feature.append(each.count(feature)) #這邊修正
            else:
                column_feature.append(0)
        column_feature = np.array(column_feature).transpose()
        Feature_Table.insert(0,feature,column_feature)
        column_feature = []
    return Feature_Table
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

#Previous day of Close ----------------------------------------------
def previous_close(vix):
    tmp = [13.29]
    value = vix.values()
    for each in value:
        if len(tmp) != len(value):
            tmp.append(each) #2014最後一個交易日的收盤價
        else:
            break
    return tmp
#--------------------------------------------------------------------

#ups and downs for VIX ----------------------------------------------
def get_modified_up_down_VIX(Table):
    
    Up_Down=[]
    for i in range(len(Table)):
        if i>=1 and i <= len(Table):
            if Table['VIX'][i] > 1.01*Table['VIX'][i-1]: # Table['VIX'][i-1]*1.01: #原本是 +0.25
                Up_Down.append(1)
            else:
                Up_Down.append(0)
        else:
            Up_Down.append(0)
    VIX_Up_Down = np.array(Up_Down).transpose()
    Table.insert(0,'Up_Down',VIX_Up_Down)
    return Table
#--------------------------------------------------------------------


#ups, holds and downs for VIX ---------------------------------------
def get_modified_up_hold_down_VIX(Table):
    
    Up_Hold_Down=[]
    for i in range(len(Table)):
        if i>=1 and i <= len(Table):
            if Table['VIX'][i] > Table['VIX'][i-1]*1.01:
                Up_Hold_Down.append(1)
            elif Table['VIX'][i] < Table['VIX'][i-1]*1.01: #
                Up_Hold_Down.append(-1)
            else:
                Up_Hold_Down.append(0)
        else:
            Up_Hold_Down.append(0)
    VIX_Up_Hold_Down = np.array(Up_Hold_Down).transpose()
    Table.insert(0,'Up_Hold_Down',VIX_Up_Hold_Down)
    return Table
#--------------------------------------------------------------------

# first order Difference(Vt - Vt-1) ---------------------------------

def diff(Table):
    difference = []
    for i in range(0,len(Table)):
        difference.append(Table['VIX'][i] - Table['LastClose'][i]) # 今收- 昨收
    return difference
#--------------------------------------------------------------------

# import other relevant numeric data
gld = pd.read_csv('/Users/renzhengzhong/Desktop/GLD.csv',index_col = 0)
sp500 = pd.read_csv('/Users/renzhengzhong/Desktop/SP500.csv',index_col = 0)
ndaq = pd.read_csv('/Users/renzhengzhong/Desktop/NDAQ.csv',index_col = 0)
dj = pd.read_csv('/Users/renzhengzhong/Desktop/DJ.csv',index_col = 0)
us5y = pd.read_csv('/Users/renzhengzhong/Desktop/US5y.csv',index_col = 0)



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
Table.insert(0,'VIX',vix.values()) #add vix(#3169) into the dataframe
Table = get_modified_up_hold_down_VIX(Table)
Table = get_modified_up_down_VIX(Table)

cate = word_category(pos, neg, unc, strong, litigious)
stem,lemm = wd_stem_lemma(cate) 
cate_table = category_feature_matrix(cate, bow) #未經過stem 或lemma的
cate_table = category_feature_matrix(stem, bow) #經過stem的 vvvvvvv
cate_table = category_feature_matrix(stem, bow) #經過lemm的
cate_table.insert(0,'LastClose',previous_close(vix))    #加入前一天收盤價
cate_table.insert(0,'VIX',vix.values())
diff = diff(cate_table)
cate_table.insert(0,'Diff',diff)

cate_table = get_modified_up_down_VIX(cate_table)
#del cate_table['Up_Down']
cate_table = get_modified_up_hold_down_VIX(cate_table)
#del cate_table['Up_Hold_Down']

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


DOC_NYT = ToCorpus(DOC_NYT)
DOC_NYT = filter_regex(DOC_NYT)
DOC_NYT = filter_stopwords(DOC_NYT)
DOC_Corpus_NYT = toCorpus(DOC_NYT)
NYT_BOW = BOW(DOC_NYT)
Date_NYT = get_Date(NYT_BOW, VIX)
bow_NYT = get_modified_BOW(NYT_BOW,Date_NYT)
vix_NYT = get_modified_VIX(VIX, Date_NYT)    
#NYT_tfidf,feature = tf_idf(bow_NYT)

cate = word_category(pos, neg, unc, strong, litigious)
stem,lemm = wd_stem_lemma(cate) 
#NYT_cate_table = category_feature_matrix(cate, bow) #未經過stem 或lemma的
NYT_cate_table = category_feature_matrix(stem, bow_NYT) #經過stem的 vvvvvvv
#NYT_cate_table = category_feature_matrix(stem, bow) #經過lemm的
NYT_cate_table.insert(0,'LastClose',previous_close(vix_NYT))    #加入前一天收盤價
NYT_cate_table.insert(0,'VIX',vix_NYT.values())
NYT_diff = diff(NYT_cate_table)
#NYT_cate_table.insert(0,'Diff',NYT_diff)  #不知為何 diff這邊有問題

NYT_cate_table = get_modified_up_down_VIX(NYT_cate_table)
#del NYT_cate_table['Up_Down']
NYT_cate_table = get_modified_up_hold_down_VIX(NYT_cate_table)
#del NYT_cate_table['Up_Hold_Down']

'''訓練與測試資料的準備'''
from sklearn import cross_validation, metrics   
# 建立訓練與測試資料
'''
train_X, test_X, train_y, test_y = \
cross_validation.train_test_split(Features_DF.iloc[:,1:5],\
                                  Features_DF.iloc[:,7],\
                                  test_size = 0.3) #choose the testing data size 
'''                               
#分類器資料
Ctrain_X, Ctest_X, Ctrain_y, Ctest_y = \
cross_validation.train_test_split(NYT_cate_table.iloc[:,4:-1],\
                                  NYT_cate_table.iloc[:,1],\
                                  test_size = 0.2) #choose the testing data size 

#回歸資料
Rtrain_X, Rtest_X, Rtrain_y, Rtest_y = \
cross_validation.train_test_split(NYT_cate_table.iloc[:,4:-1],\
                                  NYT_cate_table.iloc[:,2],\
                                  test_size = 0.2) #choose the testing data size 
                                  
#分類器資料
Ctrain_X = NYT_cate_table.iloc[0:2552,5:-1]
Ctest_X = NYT_cate_table.iloc[2552:3190,5:-1]
Ctrain_y = NYT_cate_table.iloc[0:2552:,1]
Ctest_y = NYT_cate_table.iloc[2552:3190,1]

#回歸資料
Rtrain_X = NYT_cate_table.iloc[0:2552,5:-1] #選擇'5' --> up hold down ; 選擇 4: up down
Rtest_X = NYT_cate_table.iloc[2552:3190,5:-1]
Rtrain_y = NYT_cate_table.iloc[0:2552:,2]
Rtest_y = NYT_cate_table.iloc[2552:3190,2]


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

#--------------------------------------------------------------------


