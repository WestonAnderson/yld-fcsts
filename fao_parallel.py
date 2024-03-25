#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:02:29 2021

Code to accompany the manuscript: "Preseason maize and wheat yield forecasts for early warning" by Anderson et al.

"""
import multiprocessing
import xarray as xr
import numpy as np
import pandas as pd
import _pickle as cPickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from scipy.stats import percentileofscore
import os
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning) #ignore the warnings about taking the mean of an empty slice
warnings.filterwarnings("ignore")
plt.ioff()

codePath = '/discover/nobackup/wanders7/Code/General'
savePath1 = '/discover/nobackup/wanders7/Research'
dataPath1 = '/discover/nobackup/wanders7'
dataPath2 = '/discover/nobackup/wanders7'


exec(open(codePath+'/Forecasts/forecast_functions/regions_and_seasons.py').read())
exec(open(codePath+'/Forecasts/forecast_functions/ForecastEstimate_hypergeom_wNans.py').read())
exec(open(codePath+'/Forecasts/yield_forecasts/fao/FAO_ensoYrShift.py').read())

fcstMons = [1,2,3,4,5,6,7,8,9,10,11,12]#running 1,2,3 and 4,5,6.  #[1,2,3,4,5,6,7,8,9,10,11,12]#[9,10,11,12] #[5,6,7,8] #[1,2,3,4] #halfway through

def hindcast(fcstMon):
    #=======================================#
    #      Buttons and knobs                #
    #=======================================#
    mode = 'ENSO'
    ensoTh = 0.5
    region = 'Global' #notes to include in the file name
    anomType = 'Gau' #Gau or Smooth5
    element = 'yld' #'yld' or 'prod'
    quant = (1/3)
    seasons = ['veg']#ONLY SETUP FOR  ONE SEASON AT A TIIME? I think the loops are the wrong order otherwise ['veg','plnt','har']
    pval1 = 1 #p-value 1 to use to screen for significance
    years = np.array(range(1961,2021))
    fcstYrs = years#np.arange(1961,2021)
    PSL_analogs = 'CMIP6' #['NMME','CMIP6','Obs']
    haFracTh = .5 #fraction of main cropping area that needs to be in season (not ha weighted right now)
    thresh = 0.0001 #pct of a gridcell threshold to use to define an area as cropped or not
    notes = '_50pctArea_50kgFlag_'+anomType #notes to include in the save
    #============================================#
    #       End buttons and knobs                #
    #============================================#
    #Offset to account for the difference in the definition of the first lead (first lead in the CMIP6 is lead 0, which doesn't exist in the 1900-2007 files)
    addedOffset = 1
    
    #Read in the location shape data
    with open(savePath1+'/Forecasts/YieldFCST/fao/analogFCST_objs/geo_objs/fao_locs_shps_msks.pickle', "rb") as input_file:
        locDict = cPickle.load(input_file)
       
    #Read in the location shape data
    with open(savePath1+'/Forecasts/YieldFCST/20cYlds/analogFCST_objs/geo_objs/20cYlds_locs_shps_msks.pickle', "rb") as input_file:
        locDictSub = cPickle.load(input_file)

    
    #Read in and define the crop area masks
    cMskFile = xr.open_dataset(dataPath1+'/LargeDatasets/Crop_mapping/GEOGLAM/NetCDF/allCrops_05deg.nc')
    mzCAmsk = np.copy(cMskFile['Maize_pct'].values)
    swCAmsk = np.copy(cMskFile['Spring_Wheat_pct'].values)
    wwCAmsk = np.copy(cMskFile['Winter_Wheat_pct'].values)
    mzHAmsk = np.copy(cMskFile['Maize_pct'].values)
    swHAmsk = np.copy(cMskFile['Spring_Wheat_pct'].values)
    wwHAmsk = np.copy(cMskFile['Winter_Wheat_pct'].values)
    mzCAmsk[mzCAmsk<thresh]=0; mzCAmsk[mzCAmsk>=thresh]=1
    swCAmsk[swCAmsk<thresh]=0; swCAmsk[swCAmsk>=thresh]=1
    wwCAmsk[wwCAmsk<thresh]=0; wwCAmsk[wwCAmsk>=thresh]=1
    
    
    #wheat 
    wFcstNames=[];wLocs=[];wML=[];wMLp=[];wFCST=[];wFCSTp=[] #empty wheat forecast objects
    wLeads=[];wFcstDate=[];wTargetSeas=[];wTargetYr=[];wGrowth=[];wTargAnom=[];wTargPct=[]
    wENpr=[];wNpr=[];wLNpr=[];wHarLeads=[];wTargetClimYr=[]
    #maize
    mFcstNames=[];mLocs=[];mML=[];mMLp=[];mFCST=[];mFCSTp=[] #empty maize forecast objects
    mLeads=[];mFcstDate=[];mTargetSeas=[];mTargetYr=[];mGrowth=[];mTargAnom=[];mTargPct=[]
    mENpr=[];mNpr=[];mLNpr=[];mHarLeads=[];mTargetClimYr=[]
    #loop through the months in the forecast period
    
    for iFYx in fcstYrs:
        fcstDate = str(fcstMon).zfill(2)+'.'+str(iFYx)
        if iFYx>1981:
            if PSL_analogs !='Obs':
                ensoFcst = pd.read_csv(dataPath2+'/Data/ENSO/forecasts/CPC_PSL/'+str(iFYx)+str(fcstMon).zfill(2)+\
                                         '_TimeSeries/'+str(iFYx)+str(fcstMon).zfill(2)+'_'+PSL_analogs+'_Nino3.4_Anom0p5_Probability_TimeSeries.txt',
                                         delim_whitespace=True,skiprows=12+addedOffset,nrows=22,header=None) 
                ensoFcst.columns = ['date','LN_Obs','La Niña','N_Obs','Neutral','EN_Obs','El Niño'] #name the columns
            elif PSL_analogs=='Obs':
                ensoFcst = pd.read_csv(dataPath2+'/Data/ENSO/forecasts/CPC_PSL/'+str(iFYx)+str(fcstMon).zfill(2)+\
                                         '_TimeSeries/'+str(iFYx)+str(fcstMon).zfill(2)+'_NMME_Nino3.4_Anom0p5_Probability_TimeSeries.txt',
                                         delim_whitespace=True,skiprows=12+addedOffset,nrows=22,header=None) 
                ensoFcst.columns = ['date','La Niña','La Niña_fcst','Neutral','Neutral_fcst','El Niño','El Niño_fcst'] #name the columns. Note I've switched the names so that observed category instead of forecast category will be picked up later
            ensoFcst['Season'] = ensoFcst.date.str.slice(0,3) #make columns copatible to the NMME forecast data
            ensoFcst['Season mid-month'] = np.array(range(ensoFcst.shape[0]))+1+addedOffset+fcstMon
        else:
            path = dataPath2+'/Data/ENSO/forecasts/CPC_PSL_1900-2007/forecasts/ENSO_prob_initialized_'+str(iFYx)+str(fcstMon).zfill(2)+'.dat'
            if PSL_analogs !='Obs':
                 ensoFcst = pd.read_csv(path,sep=' ',names=['date','LN_Obs','La Niña','N_Obs','Neutral','EN_Obs','El Niño'])
            elif PSL_analogs=='Obs':
                 ensoFcst = pd.read_csv(path,sep=' ',names=['date','La Niña','La Niña_fcst','Neutral','Neutral_fcst','El Niño','El Niño_fcst']) 
            ensoFcst[['La Niña','Neutral','El Niño']] = ensoFcst[['La Niña','Neutral','El Niño']]*100
            
            ensoFcst['Season'] = ensoFcst.date.str.slice(0,3) #make columns compatible to the NMME forecast data
            ensoFcst['Season mid-month'] = np.array(range(ensoFcst.shape[0]))+2+fcstMon
            
        seasEnd = {'plnt':'veg','veg':'har','har':'eos'}
        element_dict = {'yld':'yield','prod':'production'}
        
        cmap1 =  'BrBG_r'# matplotlib.colors.Colormap(Margot2_4.mpl_colormap)
        clrBr = np.arange(0,quant*2,quant/10)
        norm = Normalize(vmin=0, vmax=quant*2, clip=False)
        mapper=cm.ScalarMappable(norm=norm, cmap=cmap1)
        
        cmap2 =  'PiYG_r'# matplotlib.colors.Colormap(Margot2_4.mpl_colormap)
        clrBr2 = np.arange(0,1,20)
        norm2 = Normalize(vmin=0, vmax=1, clip=False)
        mapper2=cm.ScalarMappable(norm=norm2, cmap=cmap2)
        
        largeReg =  regExtents[region][0] # [x0,x1, y0,y1]
         
        reader = shpreader.Reader(dataPath2+'/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
        worldShps = list(reader.geometries())
        ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
        adm0names = [shape_dict.attributes['NAME'] for shape_dict in list(reader.records())] #write all shapefile country names
         
        for ixSe in seasons:
            # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ #
            #            Read in the crop calendars at a half-degree                        #
            # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ # ~ * ~ #
            ewPath = dataPath1+'/LargeDatasets/Crop_calendars/Early_Warning/NetCDF/EW_allCrops_05deg.nc'
            amisPath = dataPath1+'/LargeDatasets/Crop_calendars/AMIS/NetCDF/AMIS_allCrops_05deg.nc'
            ccEW = xr.open_dataset(ewPath)
            ccAMIS = xr.open_dataset(amisPath)
            
            lon,lat = np.meshgrid(ccEW.longitude,ccEW.latitude)
            
            ccm1 = np.copy(ccEW['Maize_1_'+ixSe].values)
            ccm2 = np.copy(ccEW['Maize_2_'+ixSe].values)
            ccwW = np.copy(ccEW['Winter_Wheat_'+ixSe].values)
            ccwS = np.copy(ccEW['Spring_Wheat_'+ixSe].values)
            ccm1[ccm1==-99] = ccAMIS['Maize_1_'+ixSe].values[ccm1==-99]
            ccm2[ccm2==-99] = ccAMIS['Maize_2_'+ixSe].values[ccm2==-99] 
            ccwW[ccwW==-99] = ccAMIS['Winter_Wheat_'+ixSe].values[ccwW==-99]
            ccwS[ccwS==-99] = ccAMIS['Spring_Wheat_'+ixSe].values[ccwS==-99]
        
            ccm1=ccm1.astype(float);ccm1[ccm1==-99]=np.nan
            ccm2=ccm2.astype(float);ccm2[ccm2==-99]=np.nan
            ccwW=ccwW.astype(float);ccwW[ccwW==-99]=np.nan
            ccwS=ccwS.astype(float);ccwS[ccwS==-99]=np.nan
        
            ccm1_end = np.copy(ccEW['Maize_1_'+seasEnd[ixSe]].values)
            ccm2_end  = np.copy(ccEW['Maize_2_'+seasEnd[ixSe]].values)
            ccwW_end  = np.copy(ccEW['Winter_Wheat_'+seasEnd[ixSe]].values)
            ccwS_end  = np.copy(ccEW['Spring_Wheat_'+seasEnd[ixSe]].values)
            ccm1_end[ccm1_end ==-99] = ccAMIS['Maize_1_'+seasEnd[ixSe]].values[ccm1_end ==-99]
            ccm2_end[ccm2_end ==-99] = ccAMIS['Maize_2_'+seasEnd[ixSe]].values[ccm2_end ==-99] 
            ccwW_end[ccwW_end ==-99] = ccAMIS['Winter_Wheat_'+seasEnd[ixSe]].values[ccwW_end ==-99]
            ccwS_end[ccwS_end ==-99] = ccAMIS['Spring_Wheat_'+seasEnd[ixSe]].values[ccwS_end ==-99]
        
            ccm1_end=ccm1_end.astype(float);ccm1_end[ccm1_end==-99]=np.nan
            ccm2_end=ccm2_end.astype(float);ccm2_end[ccm2_end==-99]=np.nan
            ccwW_end=ccwW_end.astype(float);ccwW_end[ccwW_end==-99]=np.nan
            ccwS_end=ccwS_end.astype(float);ccwS_end[ccwS_end==-99]=np.nan
            
            ccm1_end[ccm1_end<ccm1] = ccm1_end[ccm1_end<ccm1] + 365 #move the har year up when plant/harv cycle spans the calendar year
            ccm2_end[ccm2_end<ccm2] = ccm2_end[ccm2_end<ccm2] + 365 
            ccwW_end[ccwW_end<ccwW] = ccwW_end[ccwW_end<ccwW] + 365 
            ccwS_end[ccwS_end<ccwS] = ccwS_end[ccwS_end<ccwS] + 365 
            
            moDays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,
                             0+365,31+365,59+365,90+365,120+365,151+365,181+365,212+365,243+365,273+365,304+365,334+365,334+365+30])
            ccm1Mo = np.zeros([36,ccm1.shape[0],ccm1.shape[1]])
            ccm2Mo = np.zeros([36,ccm1.shape[0],ccm1.shape[1]])
            ccwWMo = np.zeros([36,ccm1.shape[0],ccm1.shape[1]])
            ccwSMo = np.zeros([36,ccm1.shape[0],ccm1.shape[1]])
            for im in range(1,25):
                ccm1Mo[im-1,(ccm1<=moDays[im])&(ccm1_end>=moDays[im])]=1
                ccm2Mo[im-1,(ccm2<=moDays[im])&(ccm2_end>=moDays[im])]=1
                ccwWMo[im-1,(ccwW<=moDays[im])&(ccwW_end>=moDays[im])]=1
                ccwSMo[im-1,(ccwS<=moDays[im])&(ccwS_end>=moDays[im])]=1
            ccm1Mo[12:24,...] = ccm1Mo[12:24,...]+ccm1Mo[:12,...] #tile the calendars
            ccm2Mo[12:24,...] = ccm2Mo[12:24,...]+ccm2Mo[:12,...]
            ccwWMo[12:24,...] = ccwWMo[12:24,...]+ccwWMo[:12,...]
            ccwSMo[12:24,...] = ccwSMo[12:24,...]+ccwSMo[:12,...]    
            ccm1Mo[24:36,...] = ccm1Mo[12:24,...]
            ccm2Mo[24:36,...] = ccm2Mo[12:24,...]
            ccwWMo[24:36,...] = ccwWMo[12:24,...]
            ccwSMo[24:36,...] = ccwSMo[12:24,...]   
            
            #add one month on either side to capture the sholder months of the climate season
            ccwWMo[1:,...] = ccwWMo[1:,...]+ccwWMo[:-1,...];ccwWMo[:-1,...] = ccwWMo[1:,...]+ccwWMo[:-1,...];ccwWMo[ccwWMo>0]=1
            ccwSMo[1:,...] = ccwSMo[1:,...]+ccwSMo[:-1,...];ccwSMo[:-1,...] = ccwSMo[1:,...]+ccwSMo[:-1,...];ccwSMo[ccwSMo>0]=1
            ccm1Mo[1:,...] = ccm1Mo[1:,...]+ccm1Mo[:-1,...];ccm1Mo[:-1,...] = ccm1Mo[1:,...]+ccm1Mo[:-1,...];ccm1Mo[ccm1Mo>0]=1
            ccm2Mo[1:,...] = ccm2Mo[1:,...]+ccm2Mo[:-1,...];ccm2Mo[:-1,...] = ccm2Mo[1:,...]+ccm2Mo[:-1,...];ccm2Mo[ccm2Mo>0]=1

            #mask to only the cropped areas
            ccm1Mo = ccm1Mo*mzCAmsk
            ccm2Mo = ccm2Mo*mzCAmsk
            ccwSMo = ccwSMo*swCAmsk
            ccwWMo = ccwWMo*wwCAmsk
            
            iLead = 0+addedOffset
            for iXseas, iXmo, iXdate in ensoFcst[['Season','Season mid-month','date']].values:
                if iLead==0+addedOffset:
                    iXmo_init=iXmo-addedOffset #track the first month to see when not to forecast
                iLead+=1
       
                ensoProbs =  ensoFcst[ensoFcst['Season mid-month'].values==iXmo][['La Niña', 'Neutral', 'El Niño']].values[0]
                ensoMxName = ['La Niña', 'Neutral', 'El Niño'][np.where(ensoProbs==np.max(ensoProbs))[0][0]]

                #read in the subnational data
                subDF = pd.read_csv(dataPath2+'/Data/crop_stats/century-long-datasets/'+
                                         'centuryCropStats_wAnoms_v2.csv')
                subDF.loc[subDF.admin0=='United States','admin0'] = 'United States of America'
                subDF.loc[subDF.admin1=='New South Wales(b)','admin1'] = 'New South Wales'
                subDF['admin'] = 'adm1_'+subDF.admin0+'_'+subDF.admin1 #create single name index
                
                subDF = subDF.loc[(((subDF.admin0=='China')&((subDF.crop=='wheat')|(subDF.crop=='maize')))|
                          ((subDF.admin0=='United States of America')&((subDF.crop=='winter wheat')|(subDF.crop=='maize')))|
                          ((subDF.admin0=='Australia')&(subDF.crop=='wheat')))] #limit to relevant crops/countries
                subDF.loc[subDF.crop=='winter wheat','crop'] = 'wheat' #rename US winter wheat as just wheat
                subDF.loc[subDF.crop=='wheat','year'] = subDF.loc[subDF.crop=='wheat','year']-1 #shift years and limit to relevant years
                subDF = subDF.loc[subDF.year.isin(years)]
                subDF = subDF[['crop','year','admin','yldAnomGauss3']]
                subDF.columns = ['crop','year','country','yldAnomGau']
                
                
                #read in the FAO data
                ensoFAOlag = "1948_2020_ensoLagged" #always use the lagged ENSO data (essentially using planting date rather than harvest)
                
                faoDFw = pd.read_csv(dataPath2+'/Data/crop_stats/FAOSTAT/processed/FAO_Wheat_'+ensoFAOlag+'_50kgFlag.csv')
                faoDFw.loc[faoDFw.QAflag==2,element+'Anom'+anomType] = np.nan#set values = nan based on flag
                faoDFw = faoDFw[faoDFw.year.isin(years)]
                yldArrW = faoDFw[['country','year',element+'Anom'+anomType]]
                #drop US, AUS, and CHN
                yldArrW = yldArrW.loc[~((yldArrW.country=='United States of America')|
                                        (yldArrW.country=='China')|
                                        (yldArrW.country=='Australia'))]
                #add subnational US, AUS, CHN
                yldArrW = pd.concat([yldArrW,subDF.loc[subDF.crop=='wheat',['year','country','yldAnomGau']]])
                yldArrW = yldArrW.set_index(['year']).pivot(columns='country')
                wDFyrs = yldArrW.index.values
                wCnts = [yldArrW.columns[x][1] for x in range(yldArrW.columns.size)]
                yldArrW = np.array(yldArrW);yldArrW = yldArrW[:,:,np.newaxis]
                
                faoDFm = pd.read_csv(dataPath2+'/Data/crop_stats/FAOSTAT/processed/FAO_Maize_'+ensoFAOlag+'_50kgFlag.csv')
                faoDFm.loc[faoDFm.QAflag==2,element+'Anom'+anomType] = np.nan
                faoDFm = faoDFm[faoDFm.year.isin(years)]
                yldArrM = faoDFm[['country','year',element+'Anom'+anomType]]
                #drop US and CHN
                yldArrM = yldArrM.loc[~((yldArrM.country=='United States of America')|
                                        (yldArrM.country=='China'))]
                #add subnational US and CHN
                yldArrM = pd.concat([yldArrM,subDF.loc[subDF.crop=='maize',['year','country','yldAnomGau']]])
                yldArrM = yldArrM.set_index(['year']).pivot(columns='country')
                mDFyrs = yldArrM.index.values
                mCnts = [yldArrM.columns[x][1] for x in range(yldArrM.columns.size)]
                yldArrM = np.array(yldArrM);yldArrM = yldArrM[:,:,np.newaxis] 
                
                for wCnt in wCnts:
                    if wCnt.split('_')[0]=='adm1':
                        wCntCnt = wCnt.split('_')[1]
                        if wCntCnt == 'United States of America':
                            subLocCnt = 'United States'
                        else: subLocCnt = wCnt.split('_')[1]
                        subDicKey = (subLocCnt, wCnt.split('_')[2])
                        
                        if not (np.isin(subDicKey,list(locDictSub.keys()))==True).all(): continue
                            #print(wCnt+' missing from loc dictionary')
                        targYr = iFYx + (iXmo-1)//12 #[ prediction year ] + [ fcst lead ] 
                        
                        ensoOffset=0 
                        if np.isin(wCntCnt,list(ensoShft['Wheat'].keys())): #If its a lag country split on the specified month
                            if (iXmo-1)%12 <= ensoShft['Wheat'][wCntCnt]:
                                ensoOffset=1 #only shift the ENSO date if BOTH you are early in the calendar year AND the growing season splits the calendar year
                                targYr = targYr - 1 #to account for the ENSO shift in the file
                        ccwMsk=ccwWMo
                        haMsk = wwHAmsk
                        msk = locDictSub[subDicKey][0]   
                    else:    
                        if np.isin(wCnt,list(locDict.keys()))==False:
                            continue
                            #print(wCnt+' missing from loc dictionary')
                        targYr = iFYx + (iXmo-1)//12 #[ prediction year ] + [ fcst lead ] 
                        
                        ensoOffset=0 
                        if np.isin(wCnt,list(ensoShft['Wheat'].keys())): #If its a lag country split on the specified month
                            if (iXmo-1)%12 <= ensoShft['Wheat'][wCnt]:
                                ensoOffset=1 #only shift the ENSO date if BOTH you are early in the calendar year AND the growing season splits the calendar year
                                targYr = targYr - 1 #to account for the ENSO shift in the file
                        if np.isin(wCnt,['Canada','Yemen']):
                            ccwMsk=ccwSMo
                            haMsk = swHAmsk
                        else: 
                            ccwMsk=ccwWMo
                            haMsk = wwHAmsk
                        msk = locDict[wCnt][0]   
                        
                    yldLoc = np.where(np.isin(wCnts,wCnt))[0][0]

                    #calculate the percentile for the observation being forecast and keep track of it for evaluation later
                    s = yldArrW[:,yldLoc,...].squeeze()
                    if np.isfinite(s).sum()<20:continue
                    pcts = [percentileofscore(s[~np.isnan(s)], x) for x in s]
                    
                    if (((iLead<=3)&((np.nansum(haMsk*(((ccwMsk[iXmo_init-3,...])*msk)>0))/np.nansum(haMsk*(msk*(ccwMsk.sum(0)>0))))>haFracTh))|
                        ((iLead<=3)&((np.nansum(haMsk*(((ccwMsk[iXmo_init-2,...])*msk)>0))/np.nansum(haMsk*(msk*(ccwMsk.sum(0)>0))))>haFracTh))): #If the cropping season in the country is already underway continue
                        continue
                    if (np.nansum(haMsk*(((ccwMsk[iXmo-1,...])*msk)>0))/np.nansum(haMsk*(msk*(ccwMsk.sum(0)>0))))>haFracTh: #check for in-season areas in the country (at least 10% to account for edge effects)
                        varProbML, mlPs, varProbFCST, fcstPs, nEvs = fcst(varArr = yldArrW[np.where(wDFyrs!=targYr)[0],yldLoc,...][:,:,np.newaxis],
                                                                            years = wDFyrs[np.where(wDFyrs!=targYr)[0]],
                                                                            ensoProbs = list(ensoProbs/100),
                                                                            quantile = quant,
                                                                            ensoSeas = iXseas,
                                                                            ensoLag = ensoOffset, 
                                                                            ensoTh = ensoTh,
                                                                            mode = mode)

                        if np.size(np.squeeze(yldArrW[np.where(wDFyrs==targYr)[0],yldLoc,...]))==0:
                            wTargAnom.append(np.nan)
                            wTargPct.append(np.nan)
                        else:
                            wTargAnom.append(np.squeeze(yldArrW[np.where(wDFyrs==targYr)[0],yldLoc,...]))
                            wTargPct.append(np.squeeze(pcts[np.where(wDFyrs==targYr)[0][0]]))

                        targClimYr = targYr + ensoOffset #to rectify ENSO shift back to climate year

                        wFcstNames.append(np.squeeze(wCnt))
                        wML.append(np.squeeze(varProbML))
                        wMLp.append(np.squeeze(mlPs))
                        wFCST.append(np.squeeze(varProbFCST))
                        wFCSTp.append(np.squeeze(fcstPs))
                        wLocs.append(np.squeeze(yldLoc))
                        wLeads.append(np.squeeze(np.repeat(iLead,mlPs.size)))
                        wFcstDate.append(np.squeeze(np.repeat(fcstDate,mlPs.size)))
                        wTargetSeas.append(np.squeeze(np.repeat(iXseas,mlPs.size)))
                        wTargetYr.append(np.squeeze(np.repeat(targYr,mlPs.size)))
                        wTargetClimYr.append(np.squeeze(np.repeat(targClimYr,mlPs.size)))
                        wGrowth.append(np.squeeze(np.repeat(ixSe,mlPs.size)))
                        wENpr.append(ensoProbs[2])
                        wNpr.append(ensoProbs[1])
                        wLNpr.append(ensoProbs[0])

                        #calculate the number of months to the end of the veg season from the current target
                        iY=0
                        wrapAround = 0
                        while((np.nansum(haMsk*((ccwMsk[iXmo-1+iY+wrapAround,...]*msk)>0)))/np.nansum(haMsk*(msk*(ccwMsk.sum(0)>0)))>haFracTh):
                            iY=iY+1
                            if iXmo+iY>=36:
                                wrapAround = -12
                        wHarLeads.append(np.squeeze(np.repeat(iLead+iY,mlPs.size)))
    
    
                for mCnt in mCnts:
                    if mCnt.split('_')[0]=='adm1':
                        mCntCnt = mCnt.split('_')[1]
                        if mCntCnt == 'United States of America':
                            subLocCnt = 'United States'
                        else: subLocCnt = mCnt.split('_')[1]
                        subDicKey = (subLocCnt, mCnt.split('_')[2])
                        
                        if not (np.isin(subDicKey,list(locDictSub.keys()))==True).all(): continue
                            #print(wCnt+' missing from loc dictionary')
                        targYr = iFYx + (iXmo-1)//12 #[ prediction year ] + [ fcst lead ] 
                        
                        ensoOffset=0 
                        if np.isin(mCntCnt,list(ensoShft['Maize'].keys())): #If its a lag country split on the specified month
                            if (iXmo-1)%12 <= ensoShft['Maize'][mCntCnt]:
                                ensoOffset=1 #only shift the ENSO date if BOTH you are early in the calendar year AND the growing season splits the calendar year
                                targYr = targYr - 1 #to account for the ENSO shift in the file
                        ccmMsk=ccm1Mo
                        haMsk = mzHAmsk
                        msk = locDictSub[subDicKey][0]   
                    else:  
                        if np.isin(mCnt,list(locDict.keys()))==False:
                            continue
                            #print(mCnt+' missing from loc dictionary')
                        targYr = iFYx + (iXmo-1)//12 #[ prediction year ] + [ fcst lead ] 
                        
                        ensoOffset=0 
                        if np.isin(mCnt,list(ensoShft['Maize'].keys())): #If its a lag country split on the specified month
                            if (iXmo-1)%12 <= ensoShft['Maize'][mCnt]:
                                ensoOffset=1 #only shift the ENSO date if BOTH you are early in the calendar year AND the growing season splits the calendar year
                                targYr = targYr - 1 #to account for the ENSO shift in the file
                        if np.isin(mCnt,['Brazil']):
                            ccmMsk=ccm2Mo
                            haMsk = mzHAmsk
                        else: 
                            ccmMsk=ccm1Mo
                            haMsk = mzHAmsk
                        msk = locDict[mCnt][0]   
                        
                    yldLoc = np.where(np.isin(mCnts,mCnt))[0][0]

                    #calculate the percentile for the observation being forecast and keep track of it for evaluation later
                    s = yldArrM[:,yldLoc,...].squeeze()
                    if np.isfinite(s).sum()<20:continue
                    pcts = [percentileofscore(s[~np.isnan(s)], x) for x in s]
                    
                    if (((iLead<=3)&((np.nansum(haMsk*(((ccmMsk[iXmo_init-3,...])*msk)>0))/np.nansum(haMsk*(msk*(ccmMsk.sum(0)>0))))>haFracTh))|
                        ((iLead<=3)&((np.nansum(haMsk*(((ccmMsk[iXmo_init-2,...])*msk)>0))/np.nansum(haMsk*(msk*(ccmMsk.sum(0)>0))))>haFracTh))): #If the cropping season in the country is already underway continue
                        continue     
                    if (np.nansum(haMsk*(((ccmMsk[iXmo-1,...])*msk)>0))/np.nansum(haMsk*(msk*(ccmMsk.sum(0)>0))))>haFracTh: #check for in-season areas in the country (at least 1% to account for edge effects)
                        varProbML, mlPs, varProbFCST, fcstPs, nEvs = fcst(varArr = yldArrM[np.where(mDFyrs!=targYr)[0],yldLoc,...][:,:,np.newaxis],
                                                                            years = mDFyrs[np.where(mDFyrs!=targYr)[0]],
                                                                            ensoProbs = list(ensoProbs/100),
                                                                            quantile = quant,
                                                                            ensoSeas = iXseas,
                                                                            ensoLag = ensoOffset, 
                                                                            ensoTh = ensoTh,
                                                                            mode = mode)


                        if np.size(np.squeeze(yldArrM[np.where(mDFyrs==targYr)[0],yldLoc,...]))==0:
                            mTargAnom.append(np.nan)
                            mTargPct.append(np.nan)
                        else:
                            mTargAnom.append(np.squeeze(yldArrM[np.where(mDFyrs==targYr)[0],yldLoc,...]))  
                            mTargPct.append(np.squeeze(pcts[np.where(mDFyrs==targYr)[0][0]]))

                        targClimYr = targYr + ensoOffset #to rectify ENSO shift back to climate year
                                                        
                        mFcstNames.append(np.squeeze(mCnt))
                        mML.append(np.squeeze(varProbML))
                        mMLp.append(np.squeeze(mlPs))
                        mFCST.append(np.squeeze(varProbFCST))
                        mFCSTp.append(np.squeeze(fcstPs))     
                        mLocs.append(np.squeeze(yldLoc))
                        mLeads.append(np.squeeze(np.repeat(iLead,mlPs.size)))
                        mFcstDate.append(np.squeeze(np.repeat(fcstDate,mlPs.size)))
                        mTargetSeas.append(np.squeeze(np.repeat(iXseas,mlPs.size)))
                        mTargetYr.append(np.squeeze(np.repeat(targYr,mlPs.size)))
                        mTargetClimYr.append(np.squeeze(np.repeat(targClimYr,mlPs.size)))
                        mGrowth.append(np.squeeze(np.repeat(ixSe,mlPs.size)))
                        mENpr.append(ensoProbs[2])
                        mNpr.append(ensoProbs[1])
                        mLNpr.append(ensoProbs[0])

                        #calculate the number of months to the end of the veg season from the current target
                        iY=0
                        wrapAround = 0
                        while((np.nansum(haMsk*(((ccmMsk[iXmo-1+iY+wrapAround,...])*msk)>0))/np.nansum(haMsk*(msk*(ccmMsk.sum(0)>0))))>haFracTh):
                            iY=iY+1
                            if iXmo+iY>=36:
                                wrapAround = -12
                        mHarLeads.append(np.squeeze(np.repeat(iLead+iY,mlPs.size)))


        #print(iFYx)
    #create dataframes, create file paths, and save forecast objects
    mData = {'location':np.array(mFcstNames),'ML':np.array(mML),'MLp':np.array(mMLp),'FCST':np.array(mFCST),'FCSTp':np.array(mFCSTp),
             'lead':np.array(mLeads),'preHar_lead':np.array(mHarLeads),'fcst_date':np.array(mFcstDate),'target_seas':np.array(mTargetSeas),
             'target_year':np.array(mTargetYr),'target_clim_year':np.array(mTargetClimYr),
             'growth_stage':np.array(mGrowth),'target_yieldAnom':np.array(mTargAnom),'target_pct':np.array(mTargPct),
             'fcstENprob':np.array(mENpr),'fcstNprob':np.array(mNpr),'fcstLNprob':np.array(mLNpr)}
    wData = {'location':np.array(wFcstNames),'ML':np.array(wML),'MLp':np.array(wMLp),'FCST':np.array(wFCST),'FCSTp':np.array(wFCSTp),
             'lead':np.array(wLeads),'preHar_lead':np.array(wHarLeads),'fcst_date':np.array(wFcstDate),'target_seas':np.array(wTargetSeas),
             'target_year':np.array(wTargetYr),'target_clim_year':np.array(wTargetClimYr),
             'growth_stage':np.array(wGrowth),'target_yieldAnom':np.array(wTargAnom),'target_pct':np.array(wTargPct),
             'fcstENprob':np.array(wENpr),'fcstNprob':np.array(wNpr),'fcstLNprob':np.array(wLNpr)}
    mDF = pd.DataFrame(data=mData)
    wDF = pd.DataFrame(data=wData)  
    mPath = savePath1+'/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/Maize/'+ixSe
    if os.path.isdir(mPath) ==False: os.mkdir(mPath)
    wPath = savePath1+'/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/Wheat/'+ixSe
    if os.path.isdir(wPath) ==False: os.mkdir(wPath)
    mDF.to_pickle(mPath+'/'+str(fcstMon).zfill(2)+str(fcstYrs[0])+'-'+str(fcstYrs[-1])+'_Maize_'+element+'_'+ixSe+'_24leads_'+
                  str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+'_'+PSL_analogs+'pslAnalogs'+'_ensoLagTrgYr'+notes+'.pkl')
    wDF.to_pickle(wPath+'/'+str(fcstMon).zfill(2)+str(fcstYrs[0])+'-'+str(fcstYrs[-1])+'_Wheat_'+element+'_'+ixSe+'_24leads_'+
                  str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+'_'+PSL_analogs+'pslAnalogs'+'_ensoLagTrgYr'+notes+'.pkl')
    
with multiprocessing.Pool() as pool:
    pool.map(hindcast,fcstMons)
    
    
