#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:41:27 2018

@author: renzhengzhong
"""


'''華爾街日報'''
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