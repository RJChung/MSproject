#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:14:29 2017

@author: renzhengzhong
"""

import re

#GET WSJ JOurnal docs
def Get_News(Dict_):
    #透過路徑開啟資料
    with open(Dict_,'r') as f:
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
    
    # 實作：找尋每則新聞index的區間為何
    startpoint = 0
    no = 0
    for each in Doc.keys():
        try: #每篇文章開頭(index1)與結尾(index1+index2)的擷取
            #i11 = re.search(' Dow Jones & Company, Inc.\)',tmp[startpoint:])
            i12 = re.search(' Dow Jones & Company, Inc.',tmp[startpoint:])
            
            #index1 = min(i11.end(),i12.end())
            index1 = i12.end()+startpoint
            #print('index1: ',index1)

            #if re.search('contributed to this article.', tmp[index1:]) != None:
             #   i21 = re.search('contributed to this article.', tmp[index1:])
            #if re.search('\(See related letter:',tmp[index1:]) != None:
             #   i22 =  re.search('\(See related letter:',tmp[index1:]) 
            if re.search('\{field\{\*fldinst',tmp[index1:]) != None:
                i23 = re.search('\{field\{\*fldinst',tmp[index1:])
            #index2 = find_Endindex(index1,i21,i22,i23)
            index2 = i23.start()+index1
    
           #print('Index2 :',index2)
            news = tmp[index1:index2]
            Doc[each] = news
            startpoint = index2
            no = no + 1
            #print(no,'startpoint: ',startpoint,'\n')
        except AttributeError:
            print('AttributeError Happend')
            print("Path.{}".format(Dict_))
            
    #格式清理
    for ID,each in Doc.items():
        try:
            each = each.replace(')','')
            each = each.replace(' All Rights Reserved.', '')
            each = each.replace('\n\n','')
            each = each.replace('pardpardeftab360ri0partightenfactor0','')
            each = each.strip()
            Doc[ID] = each
        except AttributeError:
            print('AttributeError Happend')
        
    return Doc
