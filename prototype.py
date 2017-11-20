#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:50:45 2017

@author: renzhengzhong
"""

#新聞資料處理

import os
import re

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


with open('/Users/renzhengzhong/Desktop/tidy_data/WSJ/2016/201621.rtf','r') as f:
    tmp = f.read()
#年月日表達法

if 'yyyy'  in tmp:
    #用來處理特殊的日期格式：在2015/201521 以及 2016/201621會遇到
    print('有進來ifloop')
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

startpoint = 0
index1=0
no = 0
for each in Doc.keys():
    try: #每篇文章開頭(index1)與結尾(index1+index2)的擷取
        #i11 = re.search(' Dow Jones & Company, Inc.\)',tmp[startpoint:])
        i11 = re.search(' Dow Jones & Company, Inc.',tmp[startpoint:])
        #index1 = max(i11.end(),i12.end())
        index1 = i11.end()+startpoint
        print('index1: ',index1)
        #flag1 = re.search('contributed to this article.', tmp[index1:])
        #flag2 = re.search('\(See related letter:',tmp[index1:])
        flag3 = re.search('\{field\{\*fldinst',tmp[index1:])
        
        
        #if flag1 != None:
          #  i21 = flag1
         #   print('type1 not None')
        #if flag2!= None:
          #  print('type2 not None')
         #   i22 =  flag2
        if flag3 != None:
            i23 = flag3
            print('type3 not None')

       # if flag1 and flag2 and flag3 != None:
        #    index2 = find_Endindex(index1,i21,i22,i23)
        #elif flag1 and flag3 != None:
         #   index2 = find_end(index1,i21,i23)
        #else:
        index2 = flag3.start()
        #index2 = find_Endindex(index1,i21,i22,i23)
            
        index2 = index2+index1
        news = tmp[index1:index2]
        Doc[each] = news
        startpoint = index2
        no = no + 1
        print(no,'startpoint: ',startpoint,'\n')
    except AttributeError:
        print('有發生問題')
        
#格式清理
for ID,each in Doc.items():
    each = each.replace(')','')
    each = each.replace(' All Rights Reserved.', '')
    each = each.replace('\n\n','')
    each = each.replace('pardpardeftab360ri0partightenfactor0','')
    each = each.strip()
    Doc[ID] = each
    