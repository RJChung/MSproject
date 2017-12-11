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
from nltk.collocations import *

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

from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  

def tf_idf(corpus):    
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()  
    return tfidf,word
#--------------------------------------------------------------------

#Log in Sentiment Words----------------------------------------------
def senti_word(word_path): # 預計放入四種情緒詞: pos, neg, uncertain, litigious, modal strong
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
    return ID

def BOW(dict_doc):
    bow = {}
    tmp=[]
    for ID,eachdoc in dict_doc.items():
        ID = tidy_ID(ID)
        for eachword in eachdoc:
            tmp.append(eachword)
        if bow[ID] != None:
            bow[ID] = bow[ID] + tmp
        else:
            bow[ID] = tmp
        tmp = []
    return bow
#--------------------------------------------------------------------


#Log in VIX Data ----------------------------------------------------
import pandas as pd  #Data Frame
VIX = pd.read_csv('/Users/renzhengzhong/Desktop/tidy_data/vix_test.csv')
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
WSJ_tfidf,feature = tf_idf(DOC_Corpus)
X = WSJ_tfidf.toarray()
#test data
'the' in DOC_WSJ['2017 yyyy 9 mmmm 20 dddd_1']

#--------------------------------------------------------------------



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
