#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 00:30:50 2017

@author: renzhengzhong
"""


#新聞資料處理

import os
import re

import sys
sys.path.append("/Users/renzhengzhong/Desktop/論文/Source_code/WSJReadIn.py") #讀取WSJ的function
#import WSJReadIn

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
        del Doc
    
#WSJ的所有檔案路徑
WSJ = Docpath(Path,'WSJ')
WSJYear = WSJ.getYear(Path)
WSJDoc = WSJ.getDoc(WSJYear)

#NYT的所有檔案路徑
NYT = Docpath(Path,'NYT')
NYTYear = NYT.getYear(Path)
NYTDoc = NYT.getDoc(NYTYear)

#FT的所有檔案路徑
FT = Docpath(Path,'FT')
FTYear = FT.getYear(Path)
FTDoc = FT.getDoc(FTYear)


filter_regex = re.compile(r'[^a-zA-Z0-9,.?!% ]')
Docfilter = ('{','}','\\')


#GET NYT Journal docs
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
        #dates = dates[1].replace('\n','')
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
            #print('Using min')
            return min(index2)
        elif count==1:
            #print('Using Max')
            return max(index2)
        else:
            #print('using mid')
            return mid
    
        
    def find_end(i1,i21,i23):
        index2 = [i21.start(),i23.start()]
        if min(index2)>i1:
            return min(index2)
        else:
            return max(index2)
    # 實作：找尋每則新聞index的區間為何
    startpoint = 0
    no = 0
    for each in Doc.keys():
        try: #每篇文章開頭(index1)與結尾(index1+index2)的擷取
            i11 = re.search('The New York Times Company.',tmp[startpoint:])
            index1 = i11.end()+startpoint
            flag1 = re.search('\(Photos: ', tmp[index1:])
            flag3 = re.search('f2 \'a4\'e5\'a5\'f3',tmp[index1:])
            
            if flag1 != None:
                i21 = flag1
            if flag3 != None :
                i23 = flag3
            if flag1 and flag3 != None:
                index2 = find_end(index1,i21,i23)
            elif flag3 != None:
                index2 = i23.start()
                
            index2 = index2+index1
            news = tmp[index1:index2]
            Doc[each] = news
            startpoint = index2
            no = no + 1
        except AttributeError:
            print('AttributeError Happend')
            print("Path.{}".format(Dict_))
            
    
    def Get_Final_Doc(dict_):
        final_Doc={}
        for ID, each in dict_.items():
            if len(each) >=500:
                final_Doc[ID]=each
        return final_Doc
    Doc = Get_Final_Doc(Doc)
    
    #格式清理
    for ID,each in Doc.items():
        try:
            each = each.replace(')','')
            each = each.replace(' All Rights Reserved.', '')
            each = each.replace('\n\n','')
            each = each.strip()
            Doc[ID] = each
        except AttributeError:
            print('AttributeError Happend')            

    # NYT格式清理(將Photos: 後的東西給刪除，只留下主文)
    for ID, each in Doc.items():
        flag = re.search('Photos: ',each)
        if flag != None:
            index = flag.start()
            Doc[ID] = each[0:index]
        
    return Doc

    

DOC_NYT=[]
for eachyear in NYTDoc:
    for eachfile in eachyear:
        try:
            DOC_NYT.append(Get_News(eachfile))
        except UnicodeDecodeError:
            print('Error File{}'.format(eachfile))
        except UnboundLocalError:
            print('Error File{}'.format(eachfile))
        except AttributeError:
            print('Error File{}'.format(eachfile))
