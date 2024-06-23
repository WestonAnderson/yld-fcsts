#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:31:42 2022
@author: wanders7


This script evaluates the skill of the forecasts made


"""
import numpy as np
import pandas as pd
import _pickle as cPickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import os
import cartopy.io.shapereader as shpreader
from scipy.stats import spearmanr
from sklearn import metrics
exec(open('/Users/wanders7/Documents/Code/General/forecasts/forecast_functions/TP_FP_fnct.py').read())
exec(open('/Users/wanders7/Documents/Code/General/forecasts/forecast_functions/regions_and_seasons.py').read())
#=======================================#
#      Buttons and knobs                #
#=======================================#
anomType = 'Gau'
element = 'yld' #'yld' or 'prod'
quant = (1/3)
season = 'veg'#['veg','plnt','har']
pVals = [1]
crops = ['Wheat','Maize']
region = 'Global'
analogOps = ['Obs']#['CMIP6','NMME','Obs'] #decide whether to use the CMIP 6 analogs, the NMME analogs or the observed (deterministic ENSO) forecast
fcstMons = np.arange(1,13,1)
leadEvals =  [[3,4,5,6,7,8,9],[6,7,8,9],[10,11,12,13],[14,15,16,17],[18,19,20,21],[22,23,24,25]]#
years =  np.array(range(1961,2021))#years on which the model is trained
fcstYrs = ['1961','2020'] #years for which hindcasts were made
evalType = 'singleSeas' #['lead','targSeas','best_targSeas', 'singleSeas', or 'mon_targSeas' (which referrs to both target season and month)] Determines whether estimates should be evaluated according to lead, month, or both
#IF eval type is 'both' then fcstMons is automatically np.arange(1,13,1) and leadEvals is [[i] for i in range(1,24,1)]
#IF eval type is "month" then leadEvals is list(range(1,25,1))
# "singleSeas" refers to choosing a season based on historical correlations within the cross validation framework
# "best_targSeas" chooses the highest AUC skill a posteriori (so this is not compatible with operational forecasting) and needs to be run after the 'targSeas' run
yldNotes = 'ensoLagTrgYr_50pctArea_50kgFlag_'+anomType#Keep track of the notes associated with yield processing
#============================================#
#       End buttons and knobs                #
#============================================#


seasNames = ['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ']
monNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
largeReg =  regExtents[region][0] # [x0,x1, y0,y1]
element_dict = {'yld':'yield','prod':'production'}

cmap1 =  'GnBu_r'# matplotlib.colors.Colormap(Margot2_4.mpl_colormap)
clrBr1 = np.arange(.2,.3,20)
norm1 = Normalize(vmin=0.2, vmax=0.3, clip=False)
mapper1=cm.ScalarMappable(norm=norm1, cmap=cmap1)


cmap2 =  'PiYG'# matplotlib.colors.Colormap(Margot2_4.mpl_colormap)
clrBr2 = np.arange(0.2,0.8,20)
norm2 = Normalize(vmin=0.2, vmax=0.8, clip=False)
mapper2=cm.ScalarMappable(norm=norm2, cmap=cmap2)

cmap3 =  'BrBG_r'# matplotlib.colors.Colormap(Margot2_4.mpl_colormap)
clrBr3 = np.arange(0,.66,20)
norm3 = Normalize(vmin=0, vmax=0.66, clip=False)
mapper3=cm.ScalarMappable(norm=norm3, cmap=cmap3)

#Read in the location shape data
with open(r"/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/geo_objs/fao_locs_shps_msks.pickle", "rb") as input_file:
    locDict = cPickle.load(input_file)


#============================================#
#   Plot the lead-dependent evaluation       #
#============================================#
if evalType=='lead':
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[];
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    rocFP = []; rocTP = []; rocCntry = []; rocLead = []; rocPr = []; rocCrop = []
    relFCST = []; relOBS = []; relCntry = []; relLead = []; relFcstHistY = []; relCrop=[]
    for leadTimes in leadEvals:
        for PSL_analogs in analogOps:
            for pVal in pVals:
                for crop in crops:
                    for fcstMon in fcstMons:
                        with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                  crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                  '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                  'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                            data = cPickle.load(f)
                        if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                        else: preds=pd.concat([preds,data],axis=0)
                    preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                    names = []
                    aucs = []
                    bss = []
                    #fig1 = plt.figure(111)
                    for country in np.unique(preds.location.values):
                        y_pred = preds[['FCST','target_year','target_pct']][(preds.location==country)&(preds.FCSTp<=pVal)&(np.isin(preds.preHar_lead,leadTimes))]#.groupby('target_year').mean()
                        y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                        y_yld = y_pred[['target_year','target_pct']] 
                        y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                        y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                        y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                        #limit obs to those predicted
                        if np.size(y_pred)<5:continue
                        y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                        y_true.set_index('year',inplace=True)
                        if np.sum(y_true.y_true)<1:continue
                        if np.sum(y_true.y_true==0)<1:continue
                        #limit predictions to where there are obs (e.g. no future preds)
                        y_pred = y_pred[y_pred.index.isin(y_true.index)]
                        
                        fcsts = []
                        obs_freqs = []
                        fcst_freqs = []
                        
                        fcst_prob_trhesholds = np.linspace(0,1,20)
                        width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                        for i in range(fcst_prob_trhesholds.size-1):
                            fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                        y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            fcsts.append(fcst_prob_trhesholds[i])
                            fcst_freqs.append(fcst_freq)
                            obs_freqs.append(obs_freq.values[0])
                        
                        fig = plt.figure()
                        ax1 = plt.subplot(221)
                        ax1.plot(fcsts,obs_freqs,'-o',zorder=5)
                        ax1.set_ylabel('Observed frequency')
                        ax12 = ax1.twinx()
                        hist = np.histogram(y_pred.FCST,bins=fcst_prob_trhesholds)
                        ax12.bar(x=hist[1][:-1],height=hist[0]/np.sum(hist[0]),zorder=1,color='grey',alpha=0.75,width=width[0])
                        ax12.set_ylim(0,1);ax1.set_ylim(0,1)
                        ax12.set_ylabel('Forecast frequency')
                        plt.plot([0,1],[0,1],'--',zorder=2);
                        ax1.set_title('Reliability: '+country+', \np-value threshold:'+str(pVal))
                        ax1.set_xlabel('Forecast')
                        # Containers for true positive / false positive rates
                        tp_rates = []
                        fp_rates = []
                        etses = []
                        # Define probability thresholds to use, between 0 and 1
                        probability_thresholds = np.linspace(-0.001,1.001,num=100)
                    
                        # Find true positive / false positive rate for each threshold
                        for p in probability_thresholds:
                            y_test_preds = []
                            for prob in y_pred.FCST.values:
                                if prob > p:
                                    y_test_preds.append(1)
                                else:
                                    y_test_preds.append(0)
                            tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                            ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                            etses.append(ets)
                            tp_rates.append(tp_rate)
                            fp_rates.append(fp_rate)
                        dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                        y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                        auc = np.sum(y*dx)
                        
                        bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                        
                        #plotting lists
                        names.append(country)
                        aucs.append(auc)
                        bss.append(bs)
                        
                        #ETS DF lists
                        edfCnts.append(np.repeat(country, np.size(etses)))
                        edfEts.append(etses)
                        edfPrbTh.append(probability_thresholds)
                        edfLeads.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]), np.size(etses)))
                        edfFcstMons.append(np.repeat(str(fcstMons[0])+'-'+str(fcstMons[-1]), np.size(etses)))
                        edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                        edfPvals.append(np.repeat(pVal, np.size(etses)))
                        edfCrops.append(np.repeat(crop, np.size(etses)))
                        
                        y_yld = y_yld.sort_values('year')
                        
                        #DF lists
                        dfCnts.append(country)
                        dfAucs.append(auc)
                        dfBss.append(bs)
                        dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                        dfFcstMons.append(str(fcstMons[0])+'-'+str(fcstMons[-1]))
                        dfAnalogs.append(PSL_analogs)
                        dfPvals.append(pVal)
                        dfCrops.append(crop)
                        
                        ax2 = plt.subplot(222)    
                        ax2.plot(fp_rates,tp_rates);plt.plot([0,1],[0,1],'--');
                        ax2.set_ylabel('Hit rate')
                        ax2.set_xlabel('False alarm rate')
                        ax2.set_title('ROC: '+country+', \n p-value threshold:'+str(pVal))
                        
                        ax3 = plt.subplot(212)
                        ax32 = ax3.twinx()
                        ax3.plot(y_yld.year,y_yld.target_pct-50,'k');
                        ax32.scatter(y_yld.year,-1*(y_pred-.33),s=5,color='darkorange',alpha=0.25)
                        ax32.plot(y_pred.groupby('target_year').mean().index,-1*(y_pred.groupby('target_year').mean()-.33),color='darkorange',lw=3)
                        ax3.set_title('-1*(Predicted prob of below normal) vs obs yield anom')
                        ax3.set_ylabel('Observed yield percentile',color='k')
                        ax32.set_ylabel('-1*Prob of below normal',color='darkorange')

                        fig.set_size_inches(15,10)
    
                        fig.tight_layout()
                        #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                        savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ROC_Reliability/'+crop+'/'+season
                        if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                        fig.savefig(savePath+'/'+country+'_'+element_dict[element]+'_'+season+'_'+crop+'_allMonsFcst'+\
                                    '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                    '_p'+str(pVal)+'_'+str(leadTimes[0])+'-'+str(leadTimes[-1])+'leads_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                        plt.close()
                        
                        rocFP.append(fp_rates)
                        rocTP.append(tp_rates)
                        rocCntry.append(np.repeat(country,np.size(tp_rates)))
                        rocPr.append(probability_thresholds)
                        rocLead.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]),np.size(tp_rates)))
                        rocCrop.append(np.repeat(crop,np.size(tp_rates)))
                        
                        relFCST.append(fcsts)
                        relOBS.append(obs_freqs)
                        relFcstHistY.append(hist[0]/np.sum(hist[0]))
                        relCntry.append(np.repeat(country,np.size(fcsts)))
                        relLead.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]),np.size(fcsts)))
                        relCrop.append(np.repeat(crop,np.size(fcsts)))
                        
    
                        fig = plt.figure()
                        ax = plt.subplot(111)    
                        ax.plot(probability_thresholds,etses)
                        ax.set_ylabel('Equitable Threat Score')
                        ax.set_xlabel('Probability Threshold')
                        plt.vlines(quant,-quant,1,color='grey',linestyle='--');ax.set_ylim(-quant,1)
                        plt.hlines(0,0,1,color='grey',linestyle='solid');ax.set_xlim(0,1)
                        ax.set_title('ETS: '+country+', \n p-value threshold:'+str(pVal))
                        fig.set_size_inches(15,7.5)
                        fig.tight_layout()
                        #~#~#~#~#~#~#~#~ Plot the ETS #~#~#~#~#~#~#~#~
                        savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ETS/'+crop+'/'+season
                        if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                        fig.savefig(savePath+'/ETS_'+country+'_'+element_dict[element]+'_'+season+'_'+crop+'_allMonsFcst'+\
                                    '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                    '_p'+str(pVal)+'_'+str(leadTimes[0])+'-'+str(leadTimes[-1])+'leads_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                        plt.close()
                        
                    
                    
                        
                    reader = shpreader.Reader('/Volumes/Data_Archive/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
                    worldShps = list(reader.geometries())
                    ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
                    
                    
                    # #Plot the probability differences as a map
                    # fig = plt.figure(211)
                    # ccrs.PlateCarree()
                    # ax1 = plt.subplot(211,projection=ccrs.PlateCarree());
                    # plt.title("Area under the ROC curve, all months' forecasts, leads "+str(leadTimes[0])+'-'+str(leadTimes[-1])+',\n'+\
                    #           crop+' '+season + ' season, '+str(round(quant,2))+\
                    #           ' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                    # ax1.coastlines(resolution='50m',zorder=2.5);
                    # ax1.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                    # ax1.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                    # ax1.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                    # ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                    
                    # for ixauc, ixname in list(zip(aucs,names)):
                    #     segs = locDict[ixname][1]
                    #     for ijx in range(np.size(segs)):
                    #         if np.size(segs)>1:
                    #             adm = segs[ijx]
                    #         else: adm=segs[0]
                    #         ax1.add_feature(adm, facecolor=mapper2.to_rgba(ixauc), edgecolor='k')
    
                    # ax2 = plt.subplot(212,projection=ccrs.PlateCarree());
                    # plt.title("Brier Skill Score, all months' forecasts, leads "+str(leadTimes[0])+'-'+str(leadTimes[-1])+',\n'+\
                    #           crop+' '+season + ' season, '+str(round(quant,2))+\
                    #           ' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                    # ax2.coastlines(resolution='50m',zorder=2.5);
                    # ax2.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                    # ax2.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                    # ax2.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                    # ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                    
                    # for ixbs, ixname in list(zip(bss,names)):
                    #     segs = locDict[ixname][1]
                    #     for ijx in range(np.size(segs)):
                    #         if np.size(segs)>1:
                    #             adm = segs[ijx]
                    #         else: adm=segs[0]
                    #         ax2.add_feature(adm, facecolor=mapper1.to_rgba(ixbs), edgecolor='k')
                    
                    # cbar_ax2 = fig.add_axes([0.88,0.515,0.025,0.4])
                    # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.PiYG, norm=norm2,extend='both')
                    
                    # cbar_ax2 = fig.add_axes([0.88,0.025,0.025,0.4])
                    # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.GnBu_r, norm=norm1,extend='max')
                    # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                    # fig.set_size_inches(18,12)
                    # fig.tight_layout()
                    # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                    # savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/AUC/'+crop+'/'+season
                    # if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                    # fig.savefig(savePath+'/'+element_dict[element]+'_'+season+'_'+crop+'_allMonsFcst'+\
                    #             '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                    #             '_p'+str(pVal)+'_'+str(leadTimes[0])+'-'+str(leadTimes[-1])+'leads_'+PSL_analogs+'analogs_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'.png', bbox_inches = "tight")
                    # plt.close()
    
    
    
    #create dataframes, create file paths, and save forecast objects
    fcstEtsDF = {'country':np.ravel(edfCnts),'ETS':np.ravel(edfEts),'prob_thresh':np.ravel(edfPrbTh),'leads':np.ravel(edfLeads),
                'fcst_mon':np.ravel(edfFcstMons),'PSL_Analog':np.ravel(edfAnalogs),'pval_thresh':np.ravel(edfPvals),'crop':np.ravel(edfCrops)} 
    edf = pd.DataFrame(data=fcstEtsDF)
    epath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/ETS_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    edf.to_pickle(epath)
    
    #create dataframes, create file paths, and save forecast objects
    fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'leads':np.array(dfLeads),
                'fcst_mon':np.array(dfFcstMons),'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
    df = pd.DataFrame(data=fcstEvDF)
    path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    df.to_pickle(path)

    rocDF = {'country':np.ravel(rocCntry),'crop':np.ravel(rocCrop),'leads':np.ravel(rocLead),'prob_thresh':np.ravel(rocPr),
                'FP':np.ravel(rocFP),'TP':np.ravel(rocTP)} 
    roc = pd.DataFrame(data=rocDF)
    roc_path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/ROC_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    roc.to_pickle(roc_path)
    
    relDF = {'country':np.ravel(relCntry),'crop':np.ravel(relCrop),'leads':np.ravel(relLead),
                'FcstProb':np.ravel(relFCST),'ObsFreq':np.ravel(relOBS),'histFcstFreq':np.ravel(relFcstHistY)} 
    rel = pd.DataFrame(data=relDF)
    rel_path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/Reliability_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    rel.to_pickle(rel_path)





#============================================#
#   Plot the single season evaluation       #
#============================================#
if evalType=='singleSeas':

    exec(open('/Users/wanders7/Documents/Code/General/Forecasts/yield_forecasts/fao/015_FAO_ensoYrShift.py').read())
    ensoSeasonNames = ['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ']

    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[];
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    rocFP = []; rocTP = []; rocCntry = []; rocLead = []; rocPr = []; rocCrop = []
    relFCST = []; relOBS = []; relCntry = []; relLead = []; relFcstHistY = []; relCrop=[]
    for leadTimes in leadEvals:
        for PSL_analogs in analogOps:
            for pVal in pVals:
                for crop in crops:
                    for fcstMon in fcstMons:
                        with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                  crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                  '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                  'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                            data = cPickle.load(f)
                        if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                        else: preds=pd.concat([preds,data],axis=0)
                    preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                    predsSingle = pd.DataFrame().reindex(columns=preds.columns)
                    names = []
                    aucs = []
                    bss = []
                    #fig1 = plt.figure(111)
                    for country in np.unique(preds.location.values):
                        icPreds = preds[(preds.location==country)]
                        for ixYr in icPreds.target_year.unique():
                            iyPreds = icPreds[(icPreds.target_year==ixYr)]
                            imyPreds = icPreds[(icPreds.target_year!=ixYr)][['target_year','target_pct']].dropna()
                            imyPreds = imyPreds.dropna()[~(imyPreds.dropna().duplicated())]
                            
                            mxR = 0
                            mxSeas = ''
                            for targetSeas in iyPreds.target_seas.unique():
                                iXmo = np.where(np.isin(ensoSeasonNames,targetSeas))[0][0]
                                if np.isin(country,list(ensoShft[crop].keys())): #If its a lag country split on the specified month
                                    if (iXmo-1)%12 <= ensoShft[crop][country]:                                
                                        ensoLag=1 #only shift the ENSO date if both you are early in the calendar year (before the sepcified date) AND the growing season splits the calendar year
                                else: 
                                    ensoLag=0 
                                nino = pd.read_csv('/Volumes/Data_Archive/Data/ENSO/fair_climatology/ERSSTv5_NINO34.csv')
                                nino['Year'] = nino['months since jan 1 1854']//12+1854 # add a year field
                                nino['Season'] = np.tile(ensoSeasonNames,nino.shape[0]//12) # add a season field
                                #select only the target season, and apply the lag
                                nino = nino[['Year','Anomaly']][nino.Season==targetSeas]
                                nino.columns = ['Year',targetSeas] #rename columns
                                nino['Year'] = nino['Year'].values.astype(int) - ensoLag
                                nino[targetSeas][nino[targetSeas]<-100] = np.nan #convert and drop the trailing missing values
                                nino = nino.dropna()   
                                
                                rVal = spearmanr(nino[nino.Year.isin(imyPreds.target_year)][targetSeas].values,imyPreds.target_pct.values).statistic
                                if np.abs(rVal)>mxR:
                                    mxR=np.abs(rVal)
                                    mxSeas=targetSeas
                            predsSingle=predsSingle.append(preds[(preds.location==country)&(preds.target_year==ixYr)&(preds.target_seas==mxSeas)])

                        y_pred = predsSingle[['FCST','target_year','target_pct']][(predsSingle.location==country)&(predsSingle.FCSTp<=pVal)&(np.isin(predsSingle.preHar_lead,leadTimes))]#.groupby('target_year').mean()
                        y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                        y_yld = y_pred[['target_year','target_pct']] 
                        y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                        y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                        y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                        #limit obs to those predicted
                        if np.size(y_pred)<5:continue
                        y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                        y_true.set_index('year',inplace=True)
                        if np.sum(y_true.y_true)<1:continue
                        if np.sum(y_true.y_true==0)<1:continue
                        #limit predictions to where there are obs (e.g. no future preds)
                        y_pred = y_pred[y_pred.index.isin(y_true.index)]
                        
                        fcsts = []
                        obs_freqs = []
                        fcst_freqs = []
                        
                        fcst_prob_trhesholds = np.linspace(0,1,20)
                        width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                        for i in range(fcst_prob_trhesholds.size-1):
                            fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                        y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            fcsts.append(fcst_prob_trhesholds[i])
                            fcst_freqs.append(fcst_freq)
                            obs_freqs.append(obs_freq.values[0])
                        
                        fig = plt.figure()
                        ax1 = plt.subplot(221)
                        ax1.plot(fcsts,obs_freqs,'-o',zorder=5)
                        ax1.set_ylabel('Observed frequency')
                        ax12 = ax1.twinx()
                        hist = np.histogram(y_pred.FCST,bins=fcst_prob_trhesholds)
                        ax12.bar(x=hist[1][:-1],height=hist[0]/np.sum(hist[0]),zorder=1,color='grey',alpha=0.75,width=width[0])
                        ax12.set_ylim(0,1);ax1.set_ylim(0,1)
                        ax12.set_ylabel('Forecast frequency')
                        plt.plot([0,1],[0,1],'--',zorder=2);
                        ax1.set_title('Reliability: '+country+', \np-value threshold:'+str(pVal))
                        ax1.set_xlabel('Forecast')
                        # Containers for true positive / false positive rates
                        tp_rates = []
                        fp_rates = []
                        etses = []
                        # Define probability thresholds to use, between 0 and 1
                        probability_thresholds = np.linspace(-0.001,1.001,num=100)
                    
                        # Find true positive / false positive rate for each threshold
                        for p in probability_thresholds:
                            y_test_preds = []
                            for prob in y_pred.FCST.values:
                                if prob > p:
                                    y_test_preds.append(1)
                                else:
                                    y_test_preds.append(0)
                            tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                            ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                            etses.append(ets)
                            tp_rates.append(tp_rate)
                            fp_rates.append(fp_rate)
                        dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                        y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                        auc = np.sum(y*dx)
                        
                        bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                        
                        #plotting lists
                        names.append(country)
                        aucs.append(auc)
                        bss.append(bs)
                        
                        #ETS DF lists
                        edfCnts.append(np.repeat(country, np.size(etses)))
                        edfEts.append(etses)
                        edfPrbTh.append(probability_thresholds)
                        edfLeads.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]), np.size(etses)))
                        edfFcstMons.append(np.repeat(str(fcstMons[0])+'-'+str(fcstMons[-1]), np.size(etses)))
                        edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                        edfPvals.append(np.repeat(pVal, np.size(etses)))
                        edfCrops.append(np.repeat(crop, np.size(etses)))
                        
                        y_yld = y_yld.sort_values('year')
                        
                        #DF lists
                        dfCnts.append(country)
                        dfAucs.append(auc)
                        dfBss.append(bs)
                        dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                        dfFcstMons.append(str(fcstMons[0])+'-'+str(fcstMons[-1]))
                        dfAnalogs.append(PSL_analogs)
                        dfPvals.append(pVal)
                        dfCrops.append(crop)
                        
                        ax2 = plt.subplot(222)    
                        ax2.plot(fp_rates,tp_rates);plt.plot([0,1],[0,1],'--');
                        ax2.set_ylabel('Hit rate')
                        ax2.set_xlabel('False alarm rate')
                        ax2.set_title('ROC: '+country+', \n p-value threshold:'+str(pVal))
                        
                        ax3 = plt.subplot(212)
                        ax32 = ax3.twinx()
                        ax3.plot(y_yld.year,y_yld.target_pct-50,'k');
                        ax32.scatter(y_yld.year,-1*(y_pred-.33),s=5,color='darkorange',alpha=0.25)
                        ax32.plot(y_pred.groupby('target_year').mean().index,-1*(y_pred.groupby('target_year').mean()-.33),color='darkorange',lw=3)
                        ax3.set_title('-1*(Predicted prob of below normal) vs obs yield anom')
                        ax3.set_ylabel('Observed yield percentile',color='k')
                        ax32.set_ylabel('-1*Prob of below normal',color='darkorange')

                        fig.set_size_inches(15,10)
    
                        fig.tight_layout()
                        #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                        savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ROC_Reliability/'+crop+'/'+season
                        if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                        fig.savefig(savePath+'/singleSeas_'+country+'_'+element_dict[element]+'_'+season+'_'+crop+'_allMonsFcst'+\
                                    '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                    '_p'+str(pVal)+'_'+str(leadTimes[0])+'-'+str(leadTimes[-1])+'leads_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                        plt.close()
                        
                        rocFP.append(fp_rates)
                        rocTP.append(tp_rates)
                        rocCntry.append(np.repeat(country,np.size(tp_rates)))
                        rocPr.append(probability_thresholds)
                        rocLead.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]),np.size(tp_rates)))
                        rocCrop.append(np.repeat(crop,np.size(tp_rates)))
                        
                        relFCST.append(fcsts)
                        relOBS.append(obs_freqs)
                        relFcstHistY.append(hist[0]/np.sum(hist[0]))
                        relCntry.append(np.repeat(country,np.size(fcsts)))
                        relLead.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]),np.size(fcsts)))
                        relCrop.append(np.repeat(crop,np.size(fcsts)))
                        
    
                        fig = plt.figure()
                        ax = plt.subplot(111)    
                        ax.plot(probability_thresholds,etses)
                        ax.set_ylabel('Equitable Threat Score')
                        ax.set_xlabel('Probability Threshold')
                        plt.vlines(quant,-quant,1,color='grey',linestyle='--');ax.set_ylim(-quant,1)
                        plt.hlines(0,0,1,color='grey',linestyle='solid');ax.set_xlim(0,1)
                        ax.set_title('ETS: '+country+', \n p-value threshold:'+str(pVal))
                        fig.set_size_inches(15,7.5)
                        fig.tight_layout()
                        #~#~#~#~#~#~#~#~ Plot the ETS #~#~#~#~#~#~#~#~
                        savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ETS/'+crop+'/'+season
                        if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                        fig.savefig(savePath+'/singleSeas_ETS_'+country+'_'+element_dict[element]+'_'+season+'_'+crop+'_allMonsFcst'+\
                                    '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                    '_p'+str(pVal)+'_'+str(leadTimes[0])+'-'+str(leadTimes[-1])+'leads_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                        plt.close()
                        
                    
                    
                        
                    reader = shpreader.Reader('/Volumes/Data_Archive/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
                    worldShps = list(reader.geometries())
                    ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
                    
                    
                    # #Plot the probability differences as a map
                    # fig = plt.figure(211)
                    # ccrs.PlateCarree()
                    # ax1 = plt.subplot(211,projection=ccrs.PlateCarree());
                    # plt.title("Area under the ROC curve, all months' forecasts, leads "+str(leadTimes[0])+'-'+str(leadTimes[-1])+',\n'+\
                    #           crop+' '+season + ' season, '+str(round(quant,2))+\
                    #           ' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                    # ax1.coastlines(resolution='50m',zorder=2.5);
                    # ax1.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                    # ax1.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                    # ax1.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                    # ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                    
                    # for ixauc, ixname in list(zip(aucs,names)):
                    #     segs = locDict[ixname][1]
                    #     for ijx in range(np.size(segs)):
                    #         if np.size(segs)>1:
                    #             adm = segs[ijx]
                    #         else: adm=segs[0]
                    #         ax1.add_feature(adm, facecolor=mapper2.to_rgba(ixauc), edgecolor='k')
    
                    # ax2 = plt.subplot(212,projection=ccrs.PlateCarree());
                    # plt.title("Brier Skill Score, all months' forecasts, leads "+str(leadTimes[0])+'-'+str(leadTimes[-1])+',\n'+\
                    #           crop+' '+season + ' season, '+str(round(quant,2))+\
                    #           ' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                    # ax2.coastlines(resolution='50m',zorder=2.5);
                    # ax2.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                    # ax2.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                    # ax2.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                    # ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                    
                    # for ixbs, ixname in list(zip(bss,names)):
                    #     segs = locDict[ixname][1]
                    #     for ijx in range(np.size(segs)):
                    #         if np.size(segs)>1:
                    #             adm = segs[ijx]
                    #         else: adm=segs[0]
                    #         ax2.add_feature(adm, facecolor=mapper1.to_rgba(ixbs), edgecolor='k')
                    
                    # cbar_ax2 = fig.add_axes([0.88,0.515,0.025,0.4])
                    # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.PiYG, norm=norm2,extend='both')
                    
                    # cbar_ax2 = fig.add_axes([0.88,0.025,0.025,0.4])
                    # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.GnBu_r, norm=norm1,extend='max')
                    # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                    # fig.set_size_inches(18,12)
                    # fig.tight_layout()
                    # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                    # savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/AUC/'+crop+'/'+season
                    # if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                    # fig.savefig(savePath+'/singleSeas_'+element_dict[element]+'_'+season+'_'+crop+'_allMonsFcst'+\
                    #             '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                    #             '_p'+str(pVal)+'_'+str(leadTimes[0])+'-'+str(leadTimes[-1])+'leads_'+PSL_analogs+'analogs_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'.png', bbox_inches = "tight")
                    # plt.close()
        
    
    #create dataframes, create file paths, and save forecast objects
    fcstEtsDF = {'country':np.ravel(edfCnts),'ETS':np.ravel(edfEts),'prob_thresh':np.ravel(edfPrbTh),'leads':np.ravel(edfLeads),
                'fcst_mon':np.ravel(edfFcstMons),'PSL_Analog':np.ravel(edfAnalogs),'pval_thresh':np.ravel(edfPvals),'crop':np.ravel(edfCrops)} 
    edf = pd.DataFrame(data=fcstEtsDF)
    epath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/singleSeas_ETS_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    edf.to_pickle(epath)
    
    #create dataframes, create file paths, and save forecast objects
    fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'leads':np.array(dfLeads),
                'fcst_mon':np.array(dfFcstMons),'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
    df = pd.DataFrame(data=fcstEvDF)
    path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/singleSeas_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    df.to_pickle(path)

    rocDF = {'country':np.ravel(rocCntry),'crop':np.ravel(rocCrop),'leads':np.ravel(rocLead),'prob_thresh':np.ravel(rocPr),
                'FP':np.ravel(rocFP),'TP':np.ravel(rocTP)} 
    roc = pd.DataFrame(data=rocDF)
    roc_path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/singleSeas_ROC_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    roc.to_pickle(roc_path)
    
    relDF = {'country':np.ravel(relCntry),'crop':np.ravel(relCrop),'leads':np.ravel(relLead),
                'FcstProb':np.ravel(relFCST),'ObsFreq':np.ravel(relOBS),'histFcstFreq':np.ravel(relFcstHistY)} 
    rel = pd.DataFrame(data=relDF)
    rel_path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/singleSeas_Reliability_lead_dependent_skill_'+str(years[0])+'-'+str(years[-1])+'_'+yldNotes+'_'+PSL_analogs+'.pkl'
    rel.to_pickle(rel_path)





#============================================#
#   Plot the month-dependent evaluation      #
#============================================#
if evalType=='month':
    leadEvals = [list(range(1,25))]
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[]
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];

    for PSL_analogs in analogOps:
        for fcstMon in fcstMons:
            if PSL_analogs=='Obs':
                if (fcstMon>=10)&(years[-1]==2021):years[-1]=2020 #forecasts not available for all of 2020 yet so drop those obs
            for pVal in pVals:
                for crop in crops:
                    with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                              crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                              '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                              'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                        data = cPickle.load(f)
                    preds = pd.DataFrame(data)
                    preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                    names = []
                    aucs = []
                    bss=[]
                    #fig1 = plt.figure(111)
                    for country in np.unique(preds.location.values):
                        y_pred = preds[['FCST','target_year','target_pct']][(preds.location==country)&(preds.FCSTp<=pVal)]#.groupby('target_year').mean()
                        y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                        y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                        y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                        y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                        y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                        #limit obs to those predicted
                        if np.size(y_pred)<5:continue
                        y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                        y_true.set_index('year',inplace=True)
                        if np.sum(y_true.y_true)<1:continue
                        if np.sum(y_true.y_true==0)<1:continue
                        #limit predictions to where there are obs (e.g. no future preds)
                        y_pred = y_pred[y_pred.index.isin(y_true.index)]
                        
                        
                        fcsts = []
                        obs_freqs = []
                        fcst_freqs = []
                        
                        fcst_prob_trhesholds = np.linspace(0,1,20)
                        width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                        for i in range(fcst_prob_trhesholds.size-1):
                            
                            fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                        y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            fcsts.append(fcst_prob_trhesholds[i] )
                            fcst_freqs.append(fcst_freq)
                            obs_freqs.append(obs_freq)
                        
                        fig = plt.figure()
                        ax1 = plt.subplot(121)
                        ax1.plot(fcsts,obs_freqs,'-o',zorder=5)
                        ax1.set_ylabel('Observed frequency')
                        ax12 = ax1.twinx()
                        hist = np.histogram(y_pred.FCST,bins=fcst_prob_trhesholds)
                        ax12.bar(x=hist[1][:-1],height=hist[0]/np.sum(hist[0]),zorder=1,color='grey',alpha=0.75,width=width[0])
                        ax12.set_ylim(0,1);ax1.set_ylim(0,1)
                        ax12.set_ylabel('Forecast frequency')
                        plt.plot([0,1],[0,1],'--',zorder=2);
                        ax1.set_title('Reliability: '+country+', \np-value threshold:'+str(pVal))
                        ax1.set_xlabel('Forecast')
                        # Containers for true positive / false positive rates
                        tp_rates = []
                        fp_rates = []
                        etses = []
                        # Define probability thresholds to use, between 0 and 1
                        probability_thresholds = np.linspace(-0.001,1.001,num=100)
                    
                        
                        # Find true positive / false positive rate for each threshold
                        for p in probability_thresholds:
                            y_test_preds = []
                            for prob in y_pred.FCST.values:
                                if prob > p:
                                    y_test_preds.append(1)
                                else:
                                    y_test_preds.append(0)
                            tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                            ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                            etses.append(ets)
                            tp_rates.append(tp_rate)
                            fp_rates.append(fp_rate)
                        dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                        y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                        auc = np.sum(y*dx)
                        
                        bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                        
                        #plotting lists
                        names.append(country)
                        aucs.append(auc)
                        bss.append(bs)
    
                        
                        #ETS DF lists
                        edfCnts.append(np.repeat(country, np.size(etses)))
                        edfEts.append(etses)
                        edfPrbTh.append(probability_thresholds)
                        edfLeads.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]), np.size(etses)))
                        edfFcstMons.append(np.repeat(str(fcstMons[0])+'-'+str(fcstMons[-1]), np.size(etses)))
                        edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                        edfPvals.append(np.repeat(pVal, np.size(etses)))
                        edfCrops.append(np.repeat(crop, np.size(etses)))
                        
                        #DF lists
                        dfCnts.append(country)
                        dfAucs.append(auc)
                        dfBss.append(bs)
                        dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                        dfFcstMons.append(str(fcstMons[0])+'-'+str(fcstMons[-1]))
                        dfAnalogs.append(PSL_analogs)
                        dfPvals.append(pVal)
                        dfCrops.append(crop)
                        
                        ax2 = plt.subplot(122)    
                        ax2.plot(fp_rates,tp_rates);plt.plot([0,1],[0,1],'--');
                        ax2.set_ylabel('Hit rate')
                        ax2.set_xlabel('False alarm rate')
                        ax2.set_title('ROC: '+country+', \n p-value threshold:'+str(pVal))
                    
                        fig.set_size_inches(15,7.5)
                        fig.tight_layout()
                        #~#~#~#~#~#~#~#~ Plot the ROC curve #~#~#~#~#~#~#~#~
                        savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ROC_Reliability/'+crop+'/'+season
                        if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                        fig.savefig(savePath+'/'+country+'_'+element_dict[element]+'_'+season+'_'+crop+'_'+monNames[fcstMon-1]+'fcst'+\
                                    '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                    '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                        plt.close()
                        
                        fig = plt.figure()
                        ax = plt.subplot(111)  
                        ax.plot(probability_thresholds,etses)
                        plt.vlines(quant,-quant,1,color='grey',linestyle='--');ax.set_ylim(-quant,1)
                        plt.hlines(0,0,1,color='grey',linestyle='solid');ax.set_xlim(0,1)
                        ax.set_title('ETS: '+country+', \n p-value threshold:'+str(pVal))
                        ax.set_ylabel('Equitable Threat Score')
                        ax.set_xlabel('Probability Threshold')
                        ax.set_title('ETS: '+country+', \n p-value threshold:'+str(pVal))
                    
                        fig.set_size_inches(15,7.5)
                        fig.tight_layout()
                        #~#~#~#~#~#~#~#~ Plot the ETS #~#~#~#~#~#~#~#~
                        savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ETS/'+crop+'/'+season
                        if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                        fig.savefig(savePath+'/ETS_'+country+'_'+element_dict[element]+'_'+season+'_'+crop+'_'+monNames[fcstMon-1]+'fcst'+\
                                    '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                    '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                        plt.close()
                    
                        
                    reader = shpreader.Reader('/Volumes/Data_Archive/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
                    worldShps = list(reader.geometries())
                    ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
                    
                    # #Plot the probability differences as a map
                    # fig = plt.figure(211)
                    # ccrs.PlateCarree()
                    # ax1 = plt.subplot(211,projection=ccrs.PlateCarree());
                    # plt.title("Area under the ROC curve, "+monNames[fcstMon-1]+',\n'+\
                    #           crop+' '+season + ' season, '+str(round(quant,2))+\
                    #           ' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                    # ax1.coastlines(resolution='50m',zorder=2.5);
                    # ax1.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                    # ax1.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                    # ax1.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                    # ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                    
                    # for ixauc, ixname in list(zip(aucs,names)):
                    #     segs = locDict[ixname][1]
                    #     for ijx in range(np.size(segs)):
                    #         if np.size(segs)>1:
                    #             adm = segs[ijx]
                    #         else: adm=segs[0]
                    #         ax1.add_feature(adm, facecolor=mapper2.to_rgba(ixauc), edgecolor='k')
    
                    # ax2 = plt.subplot(212,projection=ccrs.PlateCarree());
                    # plt.title("Brier Skill Score, "+monNames[fcstMon-1]+',\n'+\
                    #           crop+' '+season + ' season, '+str(round(quant,2))+\
                    #           ' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                    # ax2.coastlines(resolution='50m',zorder=2.5);
                    # ax2.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                    # ax2.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                    # ax2.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                    # ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                    
                    # for ixbs, ixname in list(zip(bss,names)):
                    #     segs = locDict[ixname][1]
                    #     for ijx in range(np.size(segs)):
                    #         if np.size(segs)>1:
                    #             adm = segs[ijx]
                    #         else: adm=segs[0]
                    #         ax2.add_feature(adm, facecolor=mapper1.to_rgba(ixbs), edgecolor='k')
                    
                    # cbar_ax2 = fig.add_axes([0.88,0.515,0.025,0.4])
                    # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.PiYG, norm=norm2,extend='both')
                    
                    # cbar_ax2 = fig.add_axes([0.88,0.025,0.025,0.4])
                    # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.GnBu_r, norm=norm1,extend='max')
                    # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                    # fig.set_size_inches(18,12)
                    # fig.tight_layout()
                    # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                    # savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/AUC/'+crop+'/'+season
                    # if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                    # fig.savefig(savePath+'/'+element_dict[element]+'_'+season+'_'+crop+'_'+monNames[fcstMon-1]+'fcst'+\
                    #             '_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                    #             '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                    # plt.close()
    
    
    
    #create dataframes, create file paths, and save forecast objects
    fcstEtsDF = {'country':np.ravel(edfCnts),'ETS':np.ravel(edfEts),'prob_thresh':np.ravel(edfPrbTh),'leads':np.ravel(edfLeads),
                'fcst_mon':np.ravel(edfFcstMons),'PSL_Analog':np.ravel(edfAnalogs),'pval_thresh':np.ravel(edfPvals),'crop':np.ravel(edfCrops)} 
    edf = pd.DataFrame(data=fcstEtsDF)
    epath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/ETS_month_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
    edf.to_pickle(epath)
    
    #create dataframes, create file paths, and save forecast objects
    fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'leads':np.array(dfLeads),
                'fcst_mon':np.array(dfFcstMons),'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
    df = pd.DataFrame(data=fcstEvDF)
    path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/month_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
    df.to_pickle(path)

#==============================================================#
#   Plot the target season dependent evaluation      #
#==============================================================#
if evalType=='targSeas':
    fcstMons = np.arange(1,13,1)
    leadEvals = [[3,4,5,6,7,8,9,10,11,12,13,14]]#[[i] for i in list(range(4,33))] # [[i] for i in list(range(1,24))] + [ [i, i+1, i+2] for i in np.arange(1,25,3)] #evaluate both each month individually and months in batches of three
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[];dfTrgSeas=[]
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    for PSL_analogs in analogOps:
        for leadTimes in leadEvals:
            for seas in seasNames:
                for pVal in pVals:
                    for crop in crops:
                        for fcstMon in fcstMons:
                            with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                      crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                      '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                      'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                                data = cPickle.load(f)
                            if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                            else: preds=pd.concat([preds,data],axis=0)
                        preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                        
                        names = []
                        aucs = []
                        bss=[]
                        #fig1 = plt.figure(111)
                        for country in np.unique(preds.location.values):
                            y_pred = preds[['FCST','target_year','target_pct','target_seas']][(preds.location==country)&(preds.FCSTp<=pVal)&
                                            (np.isin(preds.preHar_lead,leadTimes))&(np.isin(preds.target_seas,seas))]#.groupby('target_year').mean()
                            y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                            trgSeas = np.unique(y_pred[['target_seas']])
                            if trgSeas.size>1:
                                print(country)
                                print(crop)
                                print(trgSeas)
                                print('more than one target season?!?')

                            y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                            y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                            y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                            y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                            #limit obs to those predicted
                            if np.size(y_pred)<5:continue
                            y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                            y_true.set_index('year',inplace=True)
                            if np.sum(y_true.y_true)<1:continue
                            if np.sum(y_true.y_true==0)<1:continue
                            #limit predictions to where there are obs (e.g. no future preds)
                            y_pred = y_pred[y_pred.index.isin(y_true.index)]
                            
                            fcsts = []
                            obs_freqs = []
                            fcst_freqs = []
                            
                            fcst_prob_trhesholds = np.linspace(0,1,20)
                            width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                            for i in range(fcst_prob_trhesholds.size-1):
                                fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                            y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                fcsts.append(fcst_prob_trhesholds[i] )
                                fcst_freqs.append(fcst_freq)
                                obs_freqs.append(obs_freq)
                            
                            # Containers for true positive / false positive rates
                            tp_rates = []
                            fp_rates = []
                            etses = []
                            # Define probability thresholds to use, between 0 and 1
                            probability_thresholds = np.linspace(-0.001,1.001,num=100)

                            # Find true positive / false positive rate for each threshold
                            for p in probability_thresholds:
                                y_test_preds = []
                                for prob in y_pred.FCST.values:
                                    if prob > p:
                                        y_test_preds.append(1)
                                    else:
                                        y_test_preds.append(0)
                                tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                                ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                                etses.append(ets)
                                tp_rates.append(tp_rate)
                                fp_rates.append(fp_rate)
                            dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                            y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                            auc = np.sum(y*dx)
                            
                            bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                            
                            #plotting lists
                            names.append(country)
                            aucs.append(auc)
                            bss.append(bs)
        
                            #ETS DF lists
                            edfCnts.append(np.repeat(country, np.size(etses)))
                            edfEts.append(etses)
                            edfPrbTh.append(probability_thresholds)
                            edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                            edfPvals.append(np.repeat(pVal, np.size(etses)))
                            edfCrops.append(np.repeat(crop, np.size(etses)))
                            
                            #DF lists
                            dfCnts.append(country)
                            dfAucs.append(auc)
                            dfBss.append(bs)
                            dfAnalogs.append(PSL_analogs)
                            dfPvals.append(pVal)
                            dfCrops.append(crop)
                            dfTrgSeas.append(trgSeas[0])
    
        #create dataframes, create file paths, and save forecast objects
        fcstEtsDF = {'country':np.ravel(edfCnts),'ETS':np.ravel(edfEts),'prob_thresh':np.ravel(edfPrbTh),
                    'PSL_Analog':np.ravel(edfAnalogs),'pval_thresh':np.ravel(edfPvals),'crop':np.ravel(edfCrops)} 
        edf = pd.DataFrame(data=fcstEtsDF)
        epath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/targSeas_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
        edf.to_pickle(epath)
        
        #create dataframes, create file paths, and save forecast objects
        fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'targSeas':np.array(dfTrgSeas),
                    'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
        df = pd.DataFrame(data=fcstEvDF)
        path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/targSeas_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
        df.to_pickle(path)
        


#==============================================================#
#   Plot the month-AND-target season dependent evaluation      #
#==============================================================#
if evalType=='mon_targSeas':
    fcstMons = np.arange(1,13,1)
    leadEvals = [[i] for i in list(range(4,33))] # [[i] for i in list(range(1,24))] + [ [i, i+1, i+2] for i in np.arange(1,25,3)] #evaluate both each month individually and months in batches of three
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[];dfTrgSeas=[]
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    for PSL_analogs in analogOps:
        for fcstMon in fcstMons:
            for leadTimes in leadEvals:
                for seas in seasNames:
                    for pVal in pVals:
                        for crop in crops:
                            with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                      crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                      '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                      'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                                data = cPickle.load(f)
                            preds = pd.DataFrame(data)
                            preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                            
                            names = []
                            aucs = []
                            bss=[]
                            #fig1 = plt.figure(111)
                            for country in np.unique(preds.location.values):
                                y_pred = preds[['FCST','target_year','target_pct','target_seas']][(preds.location==country)&(preds.FCSTp<=pVal)&
                                                (np.isin(preds.preHar_lead,leadTimes))&(np.isin(preds.target_seas,seas))]#.groupby('target_year').mean()
                                y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                                trgSeas = np.unique(y_pred[['target_seas']])
                                if trgSeas.size>1:
                                    print(country)
                                    print(crop)
                                    print(trgSeas)
                                    print('more than one target season?!?')
    
                                y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                                y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                                y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                                y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                                #limit obs to those predicted
                                if np.size(y_pred)<5:continue
                                y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                                y_true.set_index('year',inplace=True)
                                if np.sum(y_true.y_true)<1:continue
                                if np.sum(y_true.y_true==0)<1:continue
                                #limit predictions to where there are obs (e.g. no future preds)
                                y_pred = y_pred[y_pred.index.isin(y_true.index)]
                                
                                fcsts = []
                                obs_freqs = []
                                fcst_freqs = []
                                
                                fcst_prob_trhesholds = np.linspace(0,1,20)
                                width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                                for i in range(fcst_prob_trhesholds.size-1):
                                    fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                    obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                                y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                    fcsts.append(fcst_prob_trhesholds[i] )
                                    fcst_freqs.append(fcst_freq)
                                    obs_freqs.append(obs_freq)
                                
                                # Containers for true positive / false positive rates
                                tp_rates = []
                                fp_rates = []
                                etses = []
                                # Define probability thresholds to use, between 0 and 1
                                probability_thresholds = np.linspace(-0.001,1.001,num=100)
    
                                # Find true positive / false positive rate for each threshold
                                for p in probability_thresholds:
                                    y_test_preds = []
                                    for prob in y_pred.FCST.values:
                                        if prob > p:
                                            y_test_preds.append(1)
                                        else:
                                            y_test_preds.append(0)
                                    tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                                    ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                                    etses.append(ets)
                                    tp_rates.append(tp_rate)
                                    fp_rates.append(fp_rate)
                                dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                                y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                                auc = np.sum(y*dx)
                                
                                bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                                
                                #plotting lists
                                names.append(country)
                                aucs.append(auc)
                                bss.append(bs)
            
                                #ETS DF lists
                                edfCnts.append(np.repeat(country, np.size(etses)))
                                edfEts.append(etses)
                                edfPrbTh.append(probability_thresholds)
                                edfLeads.append(np.repeat(str(leadTimes[0]), np.size(etses)))
                                edfFcstMons.append(np.repeat(str(fcstMon), np.size(etses)))
                                edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                                edfPvals.append(np.repeat(pVal, np.size(etses)))
                                edfCrops.append(np.repeat(crop, np.size(etses)))
                                
                                #DF lists
                                dfCnts.append(country)
                                dfAucs.append(auc)
                                dfBss.append(bs)
                                dfLeads.append(str(leadTimes[0]))
                                dfFcstMons.append(str(fcstMon))
                                dfAnalogs.append(PSL_analogs)
                                dfPvals.append(pVal)
                                dfCrops.append(crop)
                                dfTrgSeas.append(trgSeas[0])
    
        #create dataframes, create file paths, and save forecast objects
        fcstEtsDF = {'country':np.ravel(edfCnts),'ETS':np.ravel(edfEts),'prob_thresh':np.ravel(edfPrbTh),'leads':np.ravel(edfLeads),
                    'fcst_mon':np.ravel(edfFcstMons),'PSL_Analog':np.ravel(edfAnalogs),'pval_thresh':np.ravel(edfPvals),'crop':np.ravel(edfCrops)} 
        edf = pd.DataFrame(data=fcstEtsDF)
        epath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/ETS_month_and_targSeas_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
        edf.to_pickle(epath)
        
        #create dataframes, create file paths, and save forecast objects
        fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'leads':np.array(dfLeads),'targSeas':np.array(dfTrgSeas),
                    'fcst_mon':np.array(dfFcstMons),'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
        df = pd.DataFrame(data=fcstEvDF)
        path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/month_and_targSeas_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
        df.to_pickle(path)
        


#==============================================================#
#   Plot the best target season (3-14 lead) evaluation      #
#==============================================================#
if evalType=='best_targSeas':
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[];dfTrgSeas=[];
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    for PSL_analogs in analogOps:
        #read in the skill DF to choose the target season
        with open('/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/targSeas_dependent_skill_'+\
                  str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl', "rb") as input_file:
            skdf = cPickle.load(input_file)  
        for leadTimes in leadEvals:
            for pVal in pVals:
                for crop in crops:
                    for fcstMon in fcstMons:
                        with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                  crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                  '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                  'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                            data = cPickle.load(f)
                        if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                        else: preds=pd.concat([preds,data],axis=0)
                    preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]

                    names = []
                    aucs = []
                    bss=[]
                    #fig1 = plt.figure(111)
                    for country in np.unique(preds.location.values):
                        iskdf = skdf[(skdf.country==country)&(skdf.crop==crop)][['AUC','targSeas']]
                        seas = iskdf.loc[iskdf.AUC==np.nanmax(iskdf.AUC),'targSeas'].values[0]
                        y_pred = preds[['FCST','target_year','target_pct','target_seas']][(preds.location==country)&(preds.FCSTp<=pVal)&
                                        (np.isin(preds.preHar_lead,leadTimes))&(np.isin(preds.target_seas,seas))]#.groupby('target_year').mean()
                        y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                        #limit obs to those predicted
                        if np.size(y_pred)<5:continue
                        trgSeas = seas

                        y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                        y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                        y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                        y_pred = y_pred[['FCST','target_year']].set_index('target_year')

                        y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                        y_true.set_index('year',inplace=True)
                        if np.sum(y_true.y_true)<1:continue
                        if np.sum(y_true.y_true==0)<1:continue
                        #limit predictions to where there are obs (e.g. no future preds)
                        y_pred = y_pred[y_pred.index.isin(y_true.index)]
                        
                        fcsts = []
                        obs_freqs = []
                        fcst_freqs = []
                        
                        fcst_prob_trhesholds = np.linspace(0,1,20)
                        width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                        for i in range(fcst_prob_trhesholds.size-1):
                            fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                        y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                            fcsts.append(fcst_prob_trhesholds[i] )
                            fcst_freqs.append(fcst_freq)
                            obs_freqs.append(obs_freq)
                        
                        # Containers for true positive / false positive rates
                        tp_rates = []
                        fp_rates = []
                        etses = []
                        # Define probability thresholds to use, between 0 and 1
                        probability_thresholds = np.linspace(-0.001,1.001,num=100)

                        # Find true positive / false positive rate for each threshold
                        for p in probability_thresholds:
                            y_test_preds = []
                            for prob in y_pred.FCST.values:
                                if prob > p:
                                    y_test_preds.append(1)
                                else:
                                    y_test_preds.append(0)
                            tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                            ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                            etses.append(ets)
                            tp_rates.append(tp_rate)
                            fp_rates.append(fp_rate)
                        dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                        y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                        auc = np.sum(y*dx)
                        
                        bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                        
                        #plotting lists
                        names.append(country)
                        aucs.append(auc)
                        bss.append(bs)
  
                        
                        #DF lists
                        dfCnts.append(country)
                        dfAucs.append(auc)
                        dfBss.append(bs)
                        dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                        dfFcstMons.append(str(fcstMon))
                        dfAnalogs.append(PSL_analogs)
                        dfPvals.append(pVal)
                        dfCrops.append(crop)
                        dfTrgSeas.append(trgSeas)


        
        #create dataframes, create file paths, and save forecast objects
        fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'leads':np.array(dfLeads),'targSeas':np.array(dfTrgSeas),
                    'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
        df = pd.DataFrame(data=fcstEvDF)

        path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/best_targSeas_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
        df.to_pickle(path)


#=====================================================================================#
#      Plot the ENSO dependent evaluation with some options for lead and month        #
#=====================================================================================#
if evalType=='ENSO':
    #read in ENSO values
    nino = pd.read_csv('/Volumes/Data_Archive/Data/ENSO/NinoIndex34.csv')
    nino.iloc[np.where(nino==-99)]=np.nan #set missing values to nan
    
    fcstMonsEvals = [[1,2,3,4,5,6,7,8,9,10,11,12]]
    leadEvals = [list(range(9,15))]#[list(range(3,15)),list(range(15,27)),list(range(3,9)),list(range(9,15)),list(range(15,21)),list(range(21,27))] #evaluate both each month individually and months in batches of three
    ensoThreshEvals = [0.5,-0.5]
    
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[]
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    for PSL_analogs in analogOps:
        for fcstMons in fcstMonsEvals:
            for leadTimes in leadEvals:
                for ensoThresh in ensoThreshEvals:
                    if ((abs(ensoThresh)>0.5)&(np.isin([11,12,1],fcstMons).sum()==0)):
                        continue #only evaluate the strongest ENSO events in the months when ENSO peaks. Otherwise the sample will be very small
                    for pVal in pVals:
                        for crop in crops:
                            for fcstMon in fcstMons:
                                with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                          crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                          '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                          'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                                    data = cPickle.load(f)
                                if ensoThresh<0:
                                    ensoYrs = nino['Year'][nino.iloc[:,fcstMon]<=ensoThresh]
                                elif ensoThresh>0:
                                    ensoYrs = nino['Year'][nino.iloc[:,fcstMon]>=ensoThresh]
                                data = data.iloc[np.isin(data.fcst_date.str[3:].astype(int),ensoYrs),:] #limit to the years when ENSO is active at the forecst date
                                if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                                else: preds=pd.concat([preds,data],axis=0)
                            preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                            
                            names = []
                            aucs = []
                            bss=[]
                            #fig1 = plt.figure(111)
                            for country in np.unique(preds.location.values):
                                #we have already limited the data to the correct months and ENSO states, subselect by lead, pVal, and country
                                y_pred = preds[['FCST','target_year','target_pct']][(preds.location==country)&(preds.FCSTp<=pVal)&(np.isin(preds.preHar_lead,leadTimes))]#.groupby('target_year').mean()
                                y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                                y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                                y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                                y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                                y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                                #limit obs to those predicted
                                if np.size(y_pred)<5:continue
                                y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                                y_true.set_index('year',inplace=True)
                                if np.sum(y_true.y_true)<1:continue
                                if np.sum(y_true.y_true==0)<1:continue
                                #limit predictions to where there are obs (e.g. no future preds)
                                y_pred = y_pred[y_pred.index.isin(y_true.index)]
                                
                                fcsts = []
                                obs_freqs = []
                                fcst_freqs = []
                                
                                fcst_prob_trhesholds = np.linspace(0,1,20)
                                width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                                for i in range(fcst_prob_trhesholds.size-1):
                                    fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                    obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                                y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                    fcsts.append(fcst_prob_trhesholds[i] )
                                    fcst_freqs.append(fcst_freq)
                                    obs_freqs.append(obs_freq)

                                fig = plt.figure()
                                ax1 = plt.subplot(121)
                                ax1.plot(fcsts,obs_freqs,'-o',zorder=5)
                                ax1.set_ylabel('Observed frequency')
                                ax12 = ax1.twinx()
                                hist = np.histogram(y_pred.FCST,bins=fcst_prob_trhesholds)
                                ax12.bar(x=hist[1][:-1],height=hist[0]/np.sum(hist[0]),zorder=1,color='grey',alpha=0.75,width=width[0])
                                ax12.set_ylim(0,1);ax1.set_ylim(0,1)
                                ax12.set_ylabel('Forecast frequency')
                                plt.plot([0,1],[0,1],'--',zorder=2);
                                ax1.set_title('Reliability: '+country+', \nENSO threshold:'+str(ensoThresh))
                                ax1.set_xlabel('Forecast')

                                # Containers for true positive / false positive rates
                                tp_rates = []
                                fp_rates = []
                                etses = []
                                # Define probability thresholds to use, between 0 and 1
                                probability_thresholds = np.linspace(-0.001,1.001,num=100)
    
                                # Find true positive / false positive rate for each threshold
                                for p in probability_thresholds:
                                    y_test_preds = []
                                    for prob in y_pred.FCST.values:
                                        if prob > p:
                                            y_test_preds.append(1)
                                        else:
                                            y_test_preds.append(0)
                                    tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                                    ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                                    etses.append(ets)
                                    tp_rates.append(tp_rate)
                                    fp_rates.append(fp_rate)
                                dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                                y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                                auc = np.sum(y*dx)
                                
                                bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                                
                                #plotting lists
                                names.append(country)
                                aucs.append(auc)
                                bss.append(bs)
            
                                #ETS DF lists
                                edfCnts.append(np.repeat(country, np.size(etses)))
                                edfEts.append(etses)
                                edfPrbTh.append(probability_thresholds)
                                edfLeads.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]), np.size(etses)))
                                edfFcstMons.append(np.repeat(str(fcstMons[0])+'-'+str(fcstMons[-1]), np.size(etses)))
                                edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                                edfPvals.append(np.repeat(pVal, np.size(etses)))
                                edfCrops.append(np.repeat(crop, np.size(etses)))
                                
                                #DF lists
                                dfCnts.append(country)
                                dfAucs.append(auc)
                                dfBss.append(bs)
                                dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                                dfFcstMons.append(str(fcstMons[0])+'-'+str(fcstMons[-1]))
                                dfAnalogs.append(PSL_analogs)
                                dfPvals.append(pVal)
                                dfCrops.append(crop)

                                ax2 = plt.subplot(122)    
                                ax2.plot(fp_rates,tp_rates);plt.plot([0,1],[0,1],'--');
                                ax2.set_ylabel('Hit rate')
                                ax2.set_xlabel('False alarm rate')
                                ax2.set_title('ROC: '+country+', \n'+monNames[fcstMons[0]-1]+"-"+\
                                              monNames[fcstMons[-1]-1]+', leads'+str(leadTimes[0])+"-"+str(leadTimes[-1]))
                            
                                fig.set_size_inches(15,7.5)
                                fig.tight_layout()
                                #~#~#~#~#~#~#~#~ Plot the ROC curve #~#~#~#~#~#~#~#~
                                savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ROC_Reliability/'+crop+'/'+season
                                if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                                fig.savefig(savePath+'/'+country+'_'+element_dict[element]+'_'+season+'_'+crop+"Niño3.4_"+str(ensoThresh)+\
                                            '_'+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+'fcst_leads'+\
                                        str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                            '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                                plt.close()

                            reader = shpreader.Reader('/Volumes/Data_Archive/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
                            worldShps = list(reader.geometries())
                            ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
                    
        
                            # #Plot the probability differences as a map
                            # fig = plt.figure(211)
                            # ccrs.PlateCarree()
                            # ax1 = plt.subplot(211,projection=ccrs.PlateCarree());
                            # plt.title("Area under the ROC curve, "+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+\
                            #           " Niño 3.4: "+str(ensoThresh)+',\n'+crop+' leads '+str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+\
                            #             str(round(quant,2))+' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                            # ax1.coastlines(resolution='50m',zorder=2.5);
                            # ax1.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                            # ax1.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                            # ax1.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                            # ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                            
                            # for ixauc, ixname in list(zip(aucs,names)):
                            #     segs = locDict[ixname][1]
                            #     for ijx in range(np.size(segs)):
                            #         if np.size(segs)>1:
                            #             adm = segs[ijx]
                            #         else: adm=segs[0]
                            #         ax1.add_feature(adm, facecolor=mapper2.to_rgba(ixauc), edgecolor='k')
            
                            # ax2 = plt.subplot(212,projection=ccrs.PlateCarree());
                            # plt.title("Brier Skill Score, "+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+\
                            #           " Niño 3.4: "+str(ensoThresh)+',\n'+crop+' leads '+str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+\
                            #             str(round(quant,2))+' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                            # ax2.coastlines(resolution='50m',zorder=2.5);
                            # ax2.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                            # ax2.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                            # ax2.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                            # ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                            
                            # for ixbs, ixname in list(zip(bss,names)):
                            #     segs = locDict[ixname][1]
                            #     for ijx in range(np.size(segs)):
                            #         if np.size(segs)>1:
                            #             adm = segs[ijx]
                            #         else: adm=segs[0]
                            #         ax2.add_feature(adm, facecolor=mapper1.to_rgba(ixbs), edgecolor='k')
                            
                            # cbar_ax2 = fig.add_axes([0.88,0.515,0.025,0.4])
                            # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.PiYG, norm=norm2,extend='both')
                            
                            # cbar_ax2 = fig.add_axes([0.88,0.025,0.025,0.4])
                            # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.GnBu_r, norm=norm1,extend='max')
                            # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                            # fig.set_size_inches(18,12)
                            # fig.tight_layout()
                            # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                            # savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/AUC/'+crop+'/'+season
                            # if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                            # fig.savefig(savePath+'/'+element_dict[element]+'_'+season+'_'+crop+'_'+"Niño3.4_"+str(ensoThresh)+'_'+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+'fcst_leads'+\
                            #             str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                            #             '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                            # plt.close()
            

    #create dataframes, create file paths, and save forecast objects
    fcstEtsDF = {'country':np.ravel(edfCnts),'ETS':np.ravel(edfEts),'prob_thresh':np.ravel(edfPrbTh),'leads':np.ravel(edfLeads),
                'fcst_mon':np.ravel(edfFcstMons),'PSL_Analog':np.ravel(edfAnalogs),'pval_thresh':np.ravel(edfPvals),'crop':np.ravel(edfCrops)} 
    edf = pd.DataFrame(data=fcstEtsDF)
    epath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/ETS_ENSO_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
    edf.to_pickle(epath)
    
    #create dataframes, create file paths, and save forecast objects
    fcstEvDF = {'country':np.array(dfCnts),'AUC':np.array(dfAucs),'BSS':np.array(dfBss),'leads':np.array(dfLeads),
                'fcst_mon':np.array(dfFcstMons),'PSL_Analog':np.array(dfAnalogs),'pval_thresh':np.array(dfPvals),'crop':np.array(dfCrops)} 
    df = pd.DataFrame(data=fcstEvDF)
    path = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_objs/ENSO_dependent_skill_'+str(years[0])+'-'+str(years[-1])+yldNotes+'_'+PSL_analogs+'.pkl'
    df.to_pickle(path)
    




#=====================================================================================#
#      Plot the ENSO dependent evaluation with some options for lead and month        #
#=====================================================================================#
if evalType=='ENSOfcst':
    #read in ENSO values
    nino = pd.read_csv('/Volumes/Data_Archive/Data/ENSO/NinoIndex34.csv')
    nino.iloc[np.where(nino==-99)]=np.nan #set missing values to nan
    
    fcstMonsEvals = [[1,2,3,4,5,6,7,8,9,10,11,12]]
    leadEvals = [list(range(3,9)),list(range(9,15)),list(range(15,21)),list(range(21,27)),list(range(3,15)),list(range(15,27))] #evaluate both each month individually and months in batches of three
    fcstThreshEvals = [70,65]
    
    dfCnts = []; dfAucs=[]; dfBss=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[]
    edfCnts = []; edfEts=[]; edfPrbTh=[]; edfLeads=[]; edfFcstMons=[]; edfAnalogs=[]; edfPvals=[]; edfCrops=[];
    for PSL_analogs in analogOps:
        for fcstMons in fcstMonsEvals:
            for leadTimes in leadEvals:
                for fcstThresh in fcstThreshEvals:
                    for pVal in pVals:
                        for crop in crops:
                            for fcstMon in fcstMons:
                                with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                          crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                          '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                          'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                                    data = cPickle.load(f)
                                if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                                else: preds=pd.concat([preds,data],axis=0)
                            preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                            
                            predPlot = preds[(np.isin(preds.preHar_lead,leadTimes))&(preds.FCSTp<=pVal)]
                            
                            savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/fcst_histograms/'+crop
                            
                            fig = plt.figure()
                            plt.subplot(121)
                            vlMax = np.histogram(predPlot.fcstENprob)[0].max()*1.1
                            plt.hist(predPlot.fcstENprob);
                            plt.vlines(50,0,vlMax,'k');plt.vlines(60,0,vlMax,'k');plt.vlines(75,0,vlMax,'k')
                            plt.ylim(0,vlMax)
                            plt.title('El Niño forecasts at leads '+ str(leadTimes[0])+"-"+ str(leadTimes[-1]))
                            plt.subplot(122)
                            vlMax = np.histogram(predPlot.fcstLNprob)[0].max()*1.1
                            plt.hist(predPlot.fcstLNprob);
                            plt.vlines(50,0,vlMax,'k');plt.vlines(60,0,vlMax,'k');plt.vlines(75,0,vlMax,'k')
                            plt.ylim(0,vlMax)
                            plt.title('La Niña forecasts at leads '+ str(leadTimes[0])+"-"+ str(leadTimes[-1]))
                            fig.tight_layout()
                            fig.savefig(savePath+'_'+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+'fcst_leads'+ str(leadTimes[0])+"-"+\
                                        str(leadTimes[-1])+'_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                        '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                            plt.close()
 
                            #limit to the years when ENSO is forecast with a certain likelihood
                            preds = preds[(preds['fcstENprob']>=fcstThresh)|(preds['fcstLNprob']>=fcstThresh)]
                            
                            if preds.shape[0]<20:
                                continue
                                print('not enough forecasts of '+ensoState+' with likelihood > '+str(fcstThresh))
                            
                            names = []
                            aucs = []
                            bss=[]
                            #fig1 = plt.figure(111)
                            for country in np.unique(preds.location.values):
                                #we have already limited the data to the correct months and ENSO states, subselect by lead, pVal, and country
                                y_pred = preds[['FCST','target_year','target_pct']][(preds.location==country)&(preds.FCSTp<=pVal)&(np.isin(preds.preHar_lead,leadTimes))]#.groupby('target_year').mean()
                                y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                                y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                                y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                                y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                                y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                                #limit obs to those predicted
                                if np.size(y_pred)<5:continue
                                y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                                y_true.set_index('year',inplace=True)
                                if np.sum(y_true.y_true)<1:continue
                                if np.sum(y_true.y_true==0)<1:continue
                                #limit predictions to where there are obs (e.g. no future preds)
                                y_pred = y_pred[y_pred.index.isin(y_true.index)]
    
                                fcsts = []
                                obs_freqs = []
                                fcst_freqs = []
                                
                                fcst_prob_trhesholds = np.linspace(0,1,20)
                                width = fcst_prob_trhesholds[1:]-fcst_prob_trhesholds[:-1]
                                for i in range(fcst_prob_trhesholds.size-1):
                                    fcst_freq= y_pred[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                    obs_freq = y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].sum()/\
                                                y_true[(y_pred.FCST>=fcst_prob_trhesholds[i])&(y_pred.FCST<fcst_prob_trhesholds[i+1])].size
                                    fcsts.append(fcst_prob_trhesholds[i] )
                                    fcst_freqs.append(fcst_freq)
                                    obs_freqs.append(obs_freq)
    
                                fig = plt.figure()
                                ax1 = plt.subplot(121)
                                ax1.plot(fcsts,obs_freqs,'-o',zorder=5)
                                ax1.set_ylabel('Observed frequency')
                                ax12 = ax1.twinx()
                                hist = np.histogram(y_pred.FCST,bins=fcst_prob_trhesholds)
                                ax12.bar(x=hist[1][:-1],height=hist[0]/np.sum(hist[0]),zorder=1,color='grey',alpha=0.75,width=width[0])
                                ax12.set_ylim(0,1);ax1.set_ylim(0,1)
                                ax12.set_ylabel('Forecast frequency')
                                plt.plot([0,1],[0,1],'--',zorder=2);
                                ax1.set_title('Reliability: '+country+', \nENSO fcst threshold:'+str(fcstThresh)+'%')
                                ax1.set_xlabel('Forecast')
    
                                # Containers for true positive / false positive rates
                                tp_rates = []
                                fp_rates = []
                                etses = []
                                # Define probability thresholds to use, between 0 and 1
                                probability_thresholds = np.linspace(-0.001,1.001,num=100)
    
                                # Find true positive / false positive rate for each threshold
                                for p in probability_thresholds:
                                    y_test_preds = []
                                    for prob in y_pred.FCST.values:
                                        if prob > p:
                                            y_test_preds.append(1)
                                        else:
                                            y_test_preds.append(0)
                                    tp_rate, fp_rate = calc_TP_FP_rate(np.array(y_true.y_true.values), np.array(y_test_preds))
                                    ets = calc_ETS(np.array(y_true.y_true.values), np.array(y_test_preds))
                                    etses.append(ets)
                                    tp_rates.append(tp_rate)
                                    fp_rates.append(fp_rate)
                                dx = np.array(fp_rates)[:-1]-np.array(fp_rates)[1:]
                                y = (np.array(tp_rates)[:-1]+np.array(tp_rates)[1:])/2
                                auc = np.sum(y*dx)
                                
                                bs = calc_BS(np.squeeze(y_true.values),y_pred.FCST.values)
                                
                                #plotting lists
                                names.append(country)
                                aucs.append(auc)
                                bss.append(bs)
            
                                #ETS DF lists
                                edfCnts.append(np.repeat(country, np.size(etses)))
                                edfEts.append(etses)
                                edfPrbTh.append(probability_thresholds)
                                edfLeads.append(np.repeat(str(leadTimes[0])+'-'+str(leadTimes[-1]), np.size(etses)))
                                edfFcstMons.append(np.repeat(str(fcstMons[0])+'-'+str(fcstMons[-1]), np.size(etses)))
                                edfAnalogs.append(np.repeat(PSL_analogs, np.size(etses)))
                                edfPvals.append(np.repeat(pVal, np.size(etses)))
                                edfCrops.append(np.repeat(crop, np.size(etses)))
                                
                                #DF lists
                                dfCnts.append(country)
                                dfAucs.append(auc)
                                dfBss.append(bs)
                                dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                                dfFcstMons.append(str(fcstMons[0])+'-'+str(fcstMons[-1]))
                                dfAnalogs.append(PSL_analogs)
                                dfPvals.append(pVal)
                                dfCrops.append(crop)
    
                                ax2 = plt.subplot(122)    
                                ax2.plot(fp_rates,tp_rates);plt.plot([0,1],[0,1],'--');
                                ax2.set_ylabel('Hit rate')
                                ax2.set_xlabel('False alarm rate')
                                ax2.set_title('ROC: '+country+', \n'+monNames[fcstMons[0]-1]+"-"+\
                                              monNames[fcstMons[-1]-1]+', leads'+str(leadTimes[0])+"-"+str(leadTimes[-1]))
                            
                                fig.set_size_inches(15,7.5)
                                fig.tight_layout()
                                #~#~#~#~#~#~#~#~ Plot the ROC curve #~#~#~#~#~#~#~#~
                                savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/ROC_Reliability/'+crop+'/'+season
                                if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                                fig.savefig(savePath+'/'+country+'_'+element_dict[element]+'_'+season+'_'+crop+"ensoFcst_"+str(fcstThresh)+\
                                            '_'+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+'fcst_leads'+\
                                        str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                            '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                                plt.close()
    
                            reader = shpreader.Reader('/Volumes/Data_Archive/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
                            worldShps = list(reader.geometries())
                            ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
                    
        
                            # #Plot the probability differences as a map
                            # fig = plt.figure(211)
                            # ccrs.PlateCarree()
                            # ax1 = plt.subplot(211,projection=ccrs.PlateCarree());
                            # plt.title("Area under the ROC curve, "+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+\
                            #           " ENSO fcst at prob: "+str(fcstThresh)+'%,\n'+crop+' leads '+str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+\
                            #             str(round(quant,2))+' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                            # ax1.coastlines(resolution='50m',zorder=2.5);
                            # ax1.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                            # ax1.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                            # ax1.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                            # ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                            
                            # for ixauc, ixname in list(zip(aucs,names)):
                            #     segs = locDict[ixname][1]
                            #     for ijx in range(np.size(segs)):
                            #         if np.size(segs)>1:
                            #             adm = segs[ijx]
                            #         else: adm=segs[0]
                            #         ax1.add_feature(adm, facecolor=mapper2.to_rgba(ixauc), edgecolor='k')
            
                            # ax2 = plt.subplot(212,projection=ccrs.PlateCarree());
                            # plt.title("Brier Skill Score, "+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+\
                            #           " ENSO fcst at prob: "+str(fcstThresh)+'%,\n'+crop+' leads '+str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+\
                            #             str(round(quant,2))+' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                            # ax2.coastlines(resolution='50m',zorder=2.5);
                            # ax2.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                            # ax2.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                            # ax2.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                            # ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                            
                            # for ixbs, ixname in list(zip(bss,names)):
                            #     segs = locDict[ixname][1]
                            #     for ijx in range(np.size(segs)):
                            #         if np.size(segs)>1:
                            #             adm = segs[ijx]
                            #         else: adm=segs[0]
                            #         ax2.add_feature(adm, facecolor=mapper1.to_rgba(ixbs), edgecolor='k')
                            
                            # cbar_ax2 = fig.add_axes([0.88,0.515,0.025,0.4])
                            # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.PiYG, norm=norm2,extend='both')
                            
                            # cbar_ax2 = fig.add_axes([0.88,0.025,0.025,0.4])
                            # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.GnBu_r, norm=norm1,extend='max')
                            # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                            # fig.set_size_inches(18,12)
                            # fig.tight_layout()
                            # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                            # savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/AUC/'+crop+'/'+season
                            # if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                            # fig.savefig(savePath+'/'+element_dict[element]+'_'+season+'_'+crop+'_'+"ensoFcst_"+str(fcstThresh)+'_'+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+'fcst_leads'+\
                            #             str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                            #             '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                            # plt.close()
                
    

#=====================================================================================#
#      Plot the ENSO dependent evaluation with some options for lead and month        #
#=====================================================================================#
if evalType=='ENSOmeanprb':
    #read in ENSO values
    nino = pd.read_csv('/Volumes/Data_Archive/Data/ENSO/NinoIndex34.csv')
    nino.iloc[np.where(nino==-99)]=np.nan #set missing values to nan
    
    fcstMonsEvals = [[1,2,3,4,5,6,7,8,9,10,11,12]]
    leadEvals = [list(range(3,9)),list(range(9,15)),list(range(15,21)),list(range(21,27)),list(range(3,15)),list(range(15,27))] #evaluate both each month individually and months in batches of three
    fcstThreshEvals = [75,60,50]
    
    dfCnts = []; dfoFrqs=[]; dfpFrqs=[]; dfLeads=[]; dfFcstMons=[]; dfAnalogs=[]; dfPvals=[]; dfCrops=[]
    for PSL_analogs in analogOps:
        for fcstMons in fcstMonsEvals:
            for leadTimes in leadEvals:
                for fcstThresh in fcstThreshEvals:
                    for pVal in pVals:
                        for crop in crops:
                            for ensoState in ['LN','EN']:
                                for fcstMon in fcstMons:
                                    with open('/Users/wanders7/Documents/Research/Forecasts/YieldFCST/fao/analogFCST_objs/Combined_seasons/'+
                                              crop+'/'+season+'/'+str(fcstMon).zfill(2)+fcstYrs[0]+'-'+fcstYrs[1]+'_'+crop+
                                              '_'+element+'_'+season+'_24leads_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(round(quant,2))+'_'+PSL_analogs+
                                              'pslAnalogs_'+yldNotes+'.pkl', 'rb') as f:
                                        data = cPickle.load(f)
                                    if fcstMon==fcstMons[0]: preds = pd.DataFrame(data)
                                    else: preds=pd.concat([preds,data],axis=0)
                                preds = preds[preds.fcst_date.str[3:].astype(int).isin(years)]
                                

                                #limit to the years when ENSO is forecast with a certain likelihood
                                preds = preds[preds['fcst'+ensoState+'prob']>=fcstThresh]
                                
                                if preds.shape[0]<20:
                                    continue
                                    print('not enough forecasts of '+ensoState+' with likelihood > '+str(fcstThresh))
                                
                                names = []
                                obsFrqs = []
                                prdFrqs=[]
                                #fig1 = plt.figure(111)
                                for country in np.unique(preds.location.values):
                                    #we have already limited the data to the correct months and ENSO states, subselect by lead, pVal, and country
                                    y_pred = preds[['FCST','target_year','target_pct']][(preds.location==country)&(preds.FCSTp<=pVal)&(np.isin(preds.preHar_lead,leadTimes))]#.groupby('target_year').mean()
                                    y_pred = y_pred.dropna() # keep only years with both a forecast and an observation
                                    y_yld = y_pred[['target_year','target_pct']] #preds.loc[(preds.location==country)&(preds.FCSTp<=pVal),['target_year','target_pct']].groupby('target_year').mean()
                                    y_yld['y_true'] = y_yld['target_pct'] <=(quant*100+.00001) #correct for machine precision
                                    y_yld = y_yld.reset_index().rename(columns={'target_year':'year'})
                                    y_pred = y_pred[['FCST','target_year']].set_index('target_year')
                                    #limit obs to those predicted
                                    if np.size(y_pred)<5:continue
                                    y_true = y_yld[['year','y_true']][y_yld.year.isin(y_pred.index)]*1
                                    y_true.set_index('year',inplace=True)
                                    if np.sum(y_true.y_true)<1:continue
                                    if np.sum(y_true.y_true==0)<1:continue
                                    #limit predictions to where there are obs (e.g. no future preds)
                                    y_pred = y_pred[y_pred.index.isin(y_true.index)]

                                    #For this subset, instead of the AUC calculate the average forecast and the observed frequency
                                    obsFrq = y_true.mean()[0]
                                    prdFrq = y_pred.mean()[0]
                                    
                                    #plotting lists
                                    names.append(country)
                                    obsFrqs.append(obsFrq)
                                    prdFrqs.append(prdFrq)
                                    
                                    #DF lists
                                    dfCnts.append(country)
                                    dfoFrqs.append(obsFrq)
                                    dfpFrqs.append(prdFrq)
                                    dfLeads.append(str(leadTimes[0])+'-'+str(leadTimes[-1]))
                                    dfFcstMons.append(str(fcstMons[0])+'-'+str(fcstMons[-1]))
                                    dfAnalogs.append(PSL_analogs)
                                    dfPvals.append(pVal)
                                    dfCrops.append(crop)
    
    
                                reader = shpreader.Reader('/Volumes/Data_Archive/Data/adminBoundaries/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')
                                worldShps = list(reader.geometries())
                                ADM0 = cfeature.ShapelyFeature(worldShps, ccrs.PlateCarree())
                        
            
                                # #Plot the probability differences as a map
                                # fig = plt.figure(211)
                                # ccrs.PlateCarree()
                                # ax1 = plt.subplot(211,projection=ccrs.PlateCarree());
                                # plt.title("Observed frequency of event, "+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+\
                                #           " "+ensoState+" fcst: "+str(fcstThresh)+',\n'+crop+' leads '+str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+\
                                #             str(round(quant,2))+' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                                # ax1.coastlines(resolution='50m',zorder=2.5);
                                # ax1.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                                # ax1.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                                # ax1.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                                # ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                                
                                # for ixof, ixname in list(zip(obsFrqs,names)):
                                #     segs = locDict[ixname][1]
                                #     for ijx in range(np.size(segs)):
                                #         if np.size(segs)>1:
                                #             adm = segs[ijx]
                                #         else: adm=segs[0]
                                #         ax1.add_feature(adm, facecolor=mapper3.to_rgba(ixof), edgecolor='k')
                
                                # ax2 = plt.subplot(212,projection=ccrs.PlateCarree());
                                # plt.title("Average forecast of frequency, "+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+\
                                #           " "+ensoState+" fcst: "+str(fcstThresh)+',\n'+crop+' leads '+str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+\
                                #             str(round(quant,2))+' quantile, p-value mask:'+ str(pVal) +', PSL '+PSL_analogs+' analog forecasts')
                                # ax2.coastlines(resolution='50m',zorder=2.5);
                                # ax2.add_feature(ADM0, facecolor='none', edgecolor='k',linewidth=0.5,zorder=2.3)
                                # ax2.add_feature(cartopy.feature.BORDERS,zorder=2.5)  
                                # ax2.set_extent(largeReg,crs=ccrs.PlateCarree(central_longitude=0.0))
                                # ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k',facecolor='powderblue')
                                
                                # for ixpf, ixname in list(zip(prdFrqs,names)):
                                #     segs = locDict[ixname][1]
                                #     for ijx in range(np.size(segs)):
                                #         if np.size(segs)>1:
                                #             adm = segs[ijx]
                                #         else: adm=segs[0]
                                #         ax2.add_feature(adm, facecolor=mapper3.to_rgba(ixpf), edgecolor='k')
                                
                                # cbar_ax2 = fig.add_axes([0.88,0.515,0.025,0.4])
                                # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.BrBG_r, norm=norm3,extend='max')
                                
                                # cbar_ax2 = fig.add_axes([0.88,0.025,0.025,0.4])
                                # cb2 = mpl.colorbar.ColorbarBase(cbar_ax2, orientation='vertical', cmap=cm.BrBG_r, norm=norm3,extend='max')
                                # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                                # fig.set_size_inches(18,12)
                                # fig.tight_layout()
                                # #~#~#~#~#~#~#~#~ Plot tthe pie charts #~#~#~#~#~#~#~#~
                                # savePath = '/Users/wanders7/Documents/Research/Hindcasts/YieldFCST/fao/analogFCST_skill/meanProb/'+crop+'/'+season
                                # if os.path.isdir(savePath) ==False: os.mkdir(savePath)
                                # fig.savefig(savePath+'/'+element_dict[element]+'_'+season+'_'+crop+'_'+ensoState+'_'+str(fcstThresh)+'_'+monNames[fcstMons[0]-1]+"-"+monNames[fcstMons[-1]-1]+'fcst_leads'+\
                                #             str(leadTimes[0])+"-"+str(leadTimes[-1])+'_'+str(years[0])+'-'+str(years[-1])+'_quant'+str(np.round(quant,2))+\
                                #             '_p'+str(pVal)+'_'+PSL_analogs+'analogs_'+yldNotes+'.png', bbox_inches = "tight")
                                # plt.close()
                
