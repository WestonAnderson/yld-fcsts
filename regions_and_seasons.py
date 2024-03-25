#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:26:12 2021

@author: wanders7
"""

seasDic = {'DJF':[0,1,2],'JFM':[1,2,3],'FMA':[2,3,4],
           'JFMAM':[1,2,3,4,5],
           'MAM':[3,4,5],'AMJ':[4,5,6],'MJJ':[5,6,7],'JJA':[6,7,8],
           'AMJJA':[4,5,6,7,8],
           'JAS':[7,8,9],'ASO':[8,9,10],'SON':[9,10,11],'OND':[10,11,12],'NDJ':[11,12,13],
           'NDJFM':[11,12,13,14,15],
           'ONDJFM':[10,11,12,13,14,15],
           'DJFM':[12,13,14,15],
           'NDJFMA':[11,12,13,14,15,16],
           'Jan-Sep':[1,2,3,4,5,6,7,8,9],
           'Sep-Jun':[9,10,11,12,13,14,15,16,17,18]}



regExtents = {
    'Global':[[-180.5,180.5,-45,65],[-180.5,180.5,-45,65]],

    'Africa':[[-20,55,-40,40],[-20,55,-30,20]],
    'SSA':[[-20,55,-35,25],[-20,55,-35,25]],
    'East_Africa':[[20,55,-10,25],[38,53,-5,8]],
    'West_Africa':[[-20,25,-5,25],[-20,25,-5,25]],
    'Southern_Africa':[[10,43,-35,-5],[20,40,-40,-20]],

    'Central_Asia':[[50,90,20,45],[60,72,28,38]],
    'Southeast_Asia':[[60,150,-15,40],[60,150,-15,40]],
    'East_Asia':[[60,150,-15,60],[60,150,-15,60]],
    'China':[[75,135,15,60],[75,135,15,60]],
    'India':[[65,100,5,40],[65,100,5,40]],
    
    'EU_MENA':[[-20,75,20,65],[-20,75,20,65]],
    'EU':[[-10,40,35,65],[-10,40,35,65]],
    'MENA':[[-15,60,15,60],[-15,60,15,60]],

    'South_America':[[-85,-30,-50,10],[-85,-30,-50,10]],
    'Central_America':[[-110,-60,0,20],[-110,-60,0,20]],
    'North_America':[[-135,-60,15,60],[-135,-60,15,60]],
    'Mexico':[[-135,-120,15,60],[-135,-120,15,60]],
    'United_States':[[-125,-70,25,50],[-125,-70,25,50]],
    'SW_US':[[-125,-70,25,50],[-117,-102,25,37]],
    'SESA':[[-70,-30,-50,-10],[-70,-30,-50,-10]],
    
    'Australia':[[110,155,-40,-10],[110,155,-40,-10]],
    
    'fcst_Central_Asia':[[45,95,20,45],[60,72,28,38]],
    'fcst_SESA':[[-71,-46,-50,-20],[-70,-30,-50,-10]],
    'fcst_Southern_Africa':[[15,55,-35,5],[20,40,-40,-20]],
    'fcst_West_Africa':[[-20,30,-5,20],[-25,32,2,17]],    
    'fcst_Southeast_Asia':[[70,160,-15,30],[55,175,-11,23]],
    'fcst_Central_America':[[-100,-60,5,25],[-115,-45,5,25]],    
    'fcst_EU':[[-15,38,30,55],[-15,45,35,50]],
    }


faoCntComb = {
    'Former USSR':['Azerbaijan','Armenia','Belarus','Estonia','Kazakhstan','Kyrgyzstan','Latvia',
            'Georgia','Lithuania','Republic of Moldova','Russia','Turkmenistan','Tajikistan','Ukraine','Uzbekistan'],
    
    'Yugoslavia':['Croatia','Slovenia','Bosnia and Herzegovina','Macedonia','Serbia','Montenegro'],
    
    'Czechoslovakia':['Slovakia','Czech Republic','Czechia'],
    
    'Belgium-Luxembourg':['Belgium','Luxembourg'],
    
     'Türkiye':['Turkey'],
     
     'Sudan':['Sudan','S. Sudan']}