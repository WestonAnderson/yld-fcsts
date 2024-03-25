#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:56:50 2021

@author: wanders7

varArr is a 3-D array of shape [Time, Space, Space]. The order of lat/lon is not important
years is a 1-D array of years that correspond to the [Time] dimension of varArr
quantile is a fraction, between 0 and 1, that defines the quantile to test
ensoProbs is a three-member array of the probability of [La Niña, Neutral, El Niño]
ensoSeas is the three-month season to use for ENSO event identification. This has to be a three month season, e.g. "MAM"
ensoLag is the offset to apply to the enso year. If you want to use OND -1 to identify effects on JFM 0 soil moisture, the ensoLag would = -1
Mode defines the climate mode to use to identify events. Only ENSO or Western V Gradient ("WVG") are options right now.
verbose is a flag that indicates whether the function should print out all the parameters

"""
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy import signal

codePath = '/Users/wanders7/Documents/Code/General'
savePath1 = '/Users/wanders7/Documents/Research'
dataPath1 = '/Volumes/Svalbard'
dataPath2 = '/Volumes/Data_Archive'

ensoSeasonNames = ['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ']

def fcst(varArr, years, ensoProbs, quantile=0.333333333, ensoSeas='NDJ', ensoLag=0, ensoTh=0.5, mode='ENSO',verbose=False):
    #print out some basic parameters to make sure they're correct
    if mode=='ENSO':
        # if np.nanmin(years)>=1950:
        #     #read in ENSO values
        #     nino = pd.read_csv(dataPath2+'/Data/ENSO/NinoIndex34.csv')
        #     nino = nino.drop([0]) #drop a superfluous header
        #     #select only the target season, and apply the lag
        #     nino = nino[['Year',ensoSeas]]
        #     nino['Year'] = nino['Year'].values.astype(int) - ensoLag
        #     nino = nino[nino.Year.isin(years)]
        # else:
        #     #read in ENSO values
        #     nino = pd.read_csv(dataPath2+'/Data/ENSO/nino34.long.anom.data.csv')
        #     #select only the target season, and apply the lag
        #     nino = nino[['Year',ensoSeas]]
        #     nino['Year'] = nino['Year'].values.astype(int) - ensoLag
        #     nino[ensoSeas][nino[ensoSeas]<-100] = np.nan #convert and drop the trailing missing values
        #     nino = nino.dropna()
        #     nino[ensoSeas] = signal.detrend(nino[ensoSeas]) #linearly detrend
        #     nino = nino[nino.Year.isin(years)]
        #read in ENSO values
        nino = pd.read_csv(dataPath2+'/Data/ENSO/fair_climatology/ERSSTv5_NINO34.csv')
        nino['Year'] = nino['months since jan 1 1854']//12+1854 # add a year field
        nino['Season'] = np.tile(ensoSeasonNames,nino.shape[0]//12) # add a season field
        #select only the target season, and apply the lag
        nino = nino[['Year','Anomaly']][nino.Season==ensoSeas]
        nino.columns = ['Year',ensoSeas] #rename columns
        nino['Year'] = nino['Year'].values.astype(int) - ensoLag
        nino[ensoSeas][nino[ensoSeas]<-100] = np.nan #convert and drop the trailing missing values
        nino = nino.dropna()
        nino = nino[nino.Year.isin(years)]
    elif mode == 'WVG':
            #read in ENSO values
            nino = pd.read_csv(dataPath2+'/Data/WPG/WVG_'+ensoSeas+'_clim1960-1990_HadISST.csv')
            #select only the target season, and apply the lag
            nino = nino[['Year','WVG']]
            nino['Year'] = nino['Year'].values.astype(int) - ensoLag
            nino = nino[nino.Year.isin(years)]
            nino[ensoSeas] = nino['WVG']

    if (np.sum(ensoProbs)<0.99)|(np.sum(ensoProbs)>1.01):
        print('ERROR:  ENSO probability list does not sum to 1. Items should be between 0 and 1, summing to one. \n ensoProb=[La Niña probability, Neutral probability, El Niño probability')
        return
    if np.isin(ensoSeas,['NDJ','DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'])==False:
        print('ERROR:  ENSO season needs to be a three-month seas (e.g."OND") or similar')
        return

    evYrsLN = nino[nino[ensoSeas]<=-abs(ensoTh)].Year
    evYrsN = nino[(nino[ensoSeas]<abs(ensoTh))&(nino[ensoSeas]>-abs(ensoTh))].Year
    evYrsEN = nino[nino[ensoSeas]>=abs(ensoTh)].Year       
          
    ensoProbs = np.array(ensoProbs) #make sure this is an array not a list
    if np.where(ensoProbs==np.max(ensoProbs))[0][0]==0:
        evYrs = evYrsLN
    elif np.where(ensoProbs==np.max(ensoProbs))[0][0]==1:
        evYrs = evYrsN
    elif np.where(ensoProbs==np.max(ensoProbs))[0][0]==2:
        evYrs = evYrsEN

    #identify indices and count the number of events at a location-specific level that disregards nans
    #convert years vector to an array of the size of the dataset
    idxEN = np.isfinite(varArr[np.isin(years,evYrsEN),...])
    idxN = np.isfinite(varArr[np.isin(years,evYrsN),...])
    idxLN = np.isfinite(varArr[np.isin(years,evYrsLN),...])
    idxEv = np.isfinite(varArr[np.isin(years,evYrs),...])
    enYrNum = idxEN.sum(0) 
    nYrNum = idxN.sum(0) 
    lnYrNum = idxLN.sum(0) 
    evYrNum = idxEv.sum(0) 
    
    #count non-nan years
    yrNum = np.isfinite(varArr).sum(0)
    
    if verbose==True:
        print('Quantile: '+str(quantile))
        print('ENSO seas: '+ensoSeas)
        print('ENSO lag: '+str(ensoLag))
        print('ENSO Threshold: '+str(ensoTh))
        print(evYrs)

    #Find the integer location of the chosen quantile
    quantInt = np.array(yrNum*(quantile)).astype(int)

    #sort the varialbe data along the time axis
    varSort = np.copy(varArr)
    varSort = np.sort(varSort,0)

    #Don't apply zero-inflation yet
    #mask locations where the quantile is ill defined due to zero inflation
    #varArr[:,varSort[quantInt,...]==0]=np.nan
    #varSort[:,varSort[quantInt,...]==0]=np.nan

    #calculate the probability of being below the quantile if the most likely ENSO state materializes
    varProbML = np.ones(yrNum.shape)*np.nan
    obsEvents = np.ones(yrNum.shape)*np.nan
    obsEventsFcst = np.ones(yrNum.shape)*np.nan
    mlPs = np.ones(yrNum.shape)*np.nan
    fcstPs = np.ones(yrNum.shape)*np.nan
    for idx in range(varArr.shape[1]):
        for idy in range(varArr.shape[2]):
            if yrNum[idx,idy]==0:continue
            
            varProbML[idx,idy] = np.sum(varArr[np.isin(years,evYrs),idx,idy][idxEv[:,idx,idy]] <= varSort[:yrNum[idx,idy],idx,idy][quantInt[idx,idy]]) / evYrNum[idx,idy]
            obsEvents[idx,idy] = np.sum(varArr[np.isin(years,evYrs),idx,idy][idxEv[:,idx,idy]] <= varSort[:yrNum[idx,idy],idx,idy][quantInt[idx,idy]])

            totObs = yrNum[idx,idy]
            expEvents =  quantInt[idx,idy]
            numObs = idxEv[:,idx,idy].sum()
            #calculate the probability of being below the quantile if the most likely ENSO state materializes
            mlPs[idx,idy] = hypergeom.sf(obsEvents[idx,idy]-1,totObs,expEvents,numObs)
            
            if lnYrNum>0:
                lnProb = np.sum(varArr[np.isin(years,evYrsLN),idx,idy][idxLN[:,idx,idy]] <= varSort[:yrNum[idx,idy],idx,idy][quantInt[idx,idy]]) / lnYrNum[idx,idy]
            else: lnProb=0
            if nYrNum>0:
                nProb = np.sum(varArr[np.isin(years,evYrsN),idx,idy][idxN[:,idx,idy]] <= varSort[:yrNum[idx,idy],idx,idy][quantInt[idx,idy]]) / nYrNum[idx,idy]
            else: nProb=0
            if enYrNum>0:
                enProb = np.sum(varArr[np.isin(years,evYrsEN),idx,idy][idxEN[:,idx,idy]] <= varSort[:yrNum[idx,idy],idx,idy][quantInt[idx,idy]]) / enYrNum[idx,idy]
            else: enProb=0
            obsEventsFcst[idx,idy] = ensoProbs[0]* lnProb + ensoProbs[1]*nProb + ensoProbs[2]*enProb

            fcstPs[idx,idy] = hypergeom.sf(round(numObs*obsEventsFcst[idx,idy]-1),totObs,expEvents,numObs)

    varProbFCST = obsEventsFcst
    #flip the probability for those with a decreased likelihood of being in the quantile    
    mlPs[np.array(obsEvents/numObs)<=quantile] = 1-mlPs[np.array(obsEvents/numObs)<=quantile]
    fcstPs[np.array(obsEvents/numObs)<=quantile] = 1-fcstPs[np.array(obsEvents/numObs)<=quantile]
        
    probChanceML = mlPs
    probChanceFCST = fcstPs
    
    #mask out all nan areas
    varProbML[np.isnan(np.nanmean(varArr,0))]=np.nan
    varProbFCST[np.isnan(np.nanmean(varArr,0))]=np.nan
    probChanceML[np.isnan(np.nanmean(varArr,0))]=np.nan
    probChanceFCST[np.isnan(np.nanmean(varArr,0))]=np.nan
    
    return varProbML, probChanceML, varProbFCST, probChanceFCST, np.size(evYrs)

        