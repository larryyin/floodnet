#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:12:09 2018

@author: larry
"""
#import argparse
#parser = argparse.ArgumentParser(description='Single station net')
#parser.add_argument('--feature', metavar='str', type=str,
#                    default='sur_tidall',
#                   help='Feature options: "obs", "sur", "obs_tid", "obs_tidall", "sur_tidext", "sur_tidall"')
#parser.add_argument('--xl', metavar='int', type=int,
#                   help='x_len, look-back bars')
#parser.add_argument('--yl', metavar='int', type=int,
#                   help='y_len, prediction bars')
#parser.add_argument('--station', metavar='str', type=str,
#                   help='Station name')
#parser.add_argument('--layers', metavar='int', type=int, nargs='*',
#                   help='Hidden layers and hidden units')
#args = parser.parse_args()
#
#print(args.feature)
#print(args.xl)
#print(args.yl)
#print(args.station)
#print(args.layers)

#%%
import time
start_time_total = time.time()
import pandas as pd
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import tensorflow as tf
#from functools import partial
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import sys
import shutil
import shelve

#%%
def mkdir_clean(directory):
    try:
        shutil.rmtree(directory)
        os.makedirs(directory)
    except:
        os.makedirs(directory)

#%%
#T = pd.read_csv('series/'+'obs_h'+'.csv',header=0,index_col=0,parse_dates=True)
#
#TQ_bool = T.resample('M').count()>2
#nstation = len(TQ_bool.columns)
#TQ_bool_sum = TQ_bool.sum().sort_values(ascending=False)
#TQ_bool = TQ_bool[TQ_bool_sum.index]
#TQ = TQ_bool*np.arange(nstation)
#TQ[~TQ_bool] = np.nan
#
#
#TQ_bool_sum.head()

#%%
#station = args.station
#feature = args.feature
#x_len = args.xl
#y_len = args.yl
#n_hidden = args.layers

# Test
#station = 'Rockaway_Inlet_near_Floyd_Bennett_Field_NY'
SR = pd.read_csv('stations_distbar.csv')
stations = SR.station.values
#station = stations[iStation]
nstation = len(stations)

feature = 'obs'
#x_len = 24
#y_len = 6
#n_hidden = [256,128]

#
#period = x_len+y_len

#dir0 = 'tests/interp_bar/{:s}/'.format(station)
#mkdir_clean(dir0)

#orig_stdout = sys.stdout
#f = open(dir0+'log.txt', 'w')
#sys.stdout = f
#print('Station:',station)
print('Feature:',feature)
#f.write('station: {:s}'.format(station))
#f.write('feature: {:s}'.format(feature))
#%% Interp Bar
T0 = pd.read_csv('series/'+'obs_hs'+'.csv',header=0,index_col=0,parse_dates=True)[stations]

distm = np.load('distm_data.npz')['distm']

ISTATION = range(nstation)
K = [1,2,3]
NMISS = [[0,nstation-1],[0,0],[1,nstation-1],[1,1],[2,2],[3,5],[6,10],[11,nstation-1]]
missCase = ['all','m=0','m>0','m=1','m=2','m=3to5','m=6to10','m>10']
missCaseStr = ['all','m0','m0p','m1','m2','m3t5','m6t10','m10p']

MSE = np.zeros([nstation,len(K),len(NMISS)])
RMSE = np.zeros([nstation,len(K),len(NMISS)])
COUNTV = np.zeros([nstation,len(K),len(NMISS)])
CASE = np.empty([nstation,len(K),len(NMISS)], dtype=object)

for iStation in ISTATION:
    for ik,k in enumerate(K):
        for iMiss,nMiss in enumerate(NMISS):
            case_str = 's{:02d}k{:d}{:s}'.format(iStation,k,missCaseStr[iMiss])
            isLabel = ~T0[stations[iStation]].isnull()
            
            T = T0[isLabel]
            TV = (~T.isnull()).drop(columns=[stations[iStation]])
            nNan = (~TV).sum(axis=1)
            selNan = (nMiss[0]<=nNan) & (nNan<=nMiss[1])
            countV = selNan.sum()
            
            Y = T[stations[iStation]][selNan]
            X = T.drop(columns=[stations[iStation]])[selNan]
            dist = np.delete(distm[iStation],iStation)
            
            YO = ((X*TV/(dist**k)).sum(axis=1))/((TV/(dist**k)).sum(axis=1))
            
            mse = ((YO-Y)**2).mean()
            rmse = np.sqrt(mse)
            
            CASE[iStation,ik,iMiss] = case_str
            MSE[iStation,ik,iMiss] = mse
            RMSE[iStation,ik,iMiss] = rmse
            COUNTV[iStation,ik,iMiss] = countV
            
            print(case_str,countV,rmse)

IB = {'case':CASE,'mse':MSE,'rmse':RMSE,'count':COUNTV}

#%% Interp Dir
T0 = pd.read_csv('series/'+'obs_hs'+'.csv',header=0,index_col=0,parse_dates=True)[stations]

distm = np.load('distm_data.npz')['distgm']

ISTATION = range(nstation)
K = [1,2,3]
NMISS = [[0,nstation-1],[0,0],[1,nstation-1],[1,1],[2,2],[3,5],[6,10],[11,nstation-1]]
missCase = ['all','m=0','m>0','m=1','m=2','m=3to5','m=6to10','m>10']
missCaseStr = ['all','m0','m0p','m1','m2','m3t5','m6t10','m10p']

MSE = np.zeros([nstation,len(K),len(NMISS)])
RMSE = np.zeros([nstation,len(K),len(NMISS)])
COUNTV = np.zeros([nstation,len(K),len(NMISS)])
CASE = np.empty([nstation,len(K),len(NMISS)], dtype=object)

for iStation in ISTATION:
    for ik,k in enumerate(K):
        for iMiss,nMiss in enumerate(NMISS):
            case_str = 's{:02d}k{:d}{:s}'.format(iStation,k,missCaseStr[iMiss])
            isLabel = ~T0[stations[iStation]].isnull()
            
            T = T0[isLabel]
            TV = (~T.isnull()).drop(columns=[stations[iStation]])
            nNan = (~TV).sum(axis=1)
            selNan = (nMiss[0]<=nNan) & (nNan<=nMiss[1])
            countV = selNan.sum()
            
            Y = T[stations[iStation]][selNan]
            X = T.drop(columns=[stations[iStation]])[selNan]
            dist = np.delete(distm[iStation],iStation)
            
            YO = ((X*TV/(dist**k)).sum(axis=1))/((TV/(dist**k)).sum(axis=1))
            
            mse = ((YO-Y)**2).mean()
            rmse = np.sqrt(mse)
            
            CASE[iStation,ik,iMiss] = case_str
            MSE[iStation,ik,iMiss] = mse
            RMSE[iStation,ik,iMiss] = rmse
            COUNTV[iStation,ik,iMiss] = countV
            
            print(case_str,countV,rmse)

ID = {'case':CASE,'mse':MSE,'rmse':RMSE,'count':COUNTV}

#%% MSE plot
ind = np.arange(len(NMISS))
width = 0.2 

#iStation = 0
#ik = 0
for iStation in range(nstation):
    print('Printing:',iStation)
    fig, axes = plt.subplots(1,len(K), sharey=True, figsize=(8,3))
    for ik,ax in enumerate(axes):
        rmse_ID = ID['rmse'][iStation,ik,:]
        rmse_IB = IB['rmse'][iStation,ik,:]
        
        rects1 = ax.bar(ind, rmse_ID, width, color='grey')
        rects2 = ax.bar(ind+width, rmse_IB, width, color='k')
        
        ax.set_title('k={:d}'.format(K[ik]))
        ax.set_xticks(ind + width / 2)
        #ax.set_xticklabels(missCase)
        ax.set_xticklabels(missCase,rotation=60)
        if ik==0:
            ax.set_ylabel('RMSE',weight='bold')
            ax.legend((rects1[0], rects2[0]), ('DirectDist', 'BarrierDist'), loc=2)
        ymax = np.nanmax([ID['rmse'][iStation,:,:],IB['rmse'][iStation,:,:]])
        ax.set_ylim([0,ymax*1.7])
    #    if ik==1:
    #        ax.set_xlabel('#{:d} {:s}'.format(iStation,stations[iStation]),weight='bold')
    plt.suptitle('#{:d} {:s}'.format(iStation,stations[iStation]),weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/interp/sin_{:02d}.png'.format(iStation), dpi=150)
    plt.close(fig)

#%% MSE plot
ind = np.arange(len(NMISS))
width = 0.2 

#iStation = 0
#ik = 0
for iStation in range(nstation):
    print('Printing:',iStation)
    fig, axes = plt.subplots(1,len(K), sharey=True, figsize=(8,3))
    for ik,ax in enumerate(axes):
        rmse_ID = ID['rmse'][iStation,ik,:]
        rmse_IB = IB['rmse'][iStation,ik,:]
        
        rects1 = ax.bar(ind, rmse_ID, width, color='grey')
        rects2 = ax.bar(ind+width, rmse_IB, width, color='k')
        
        ax.set_title('k={:d}'.format(K[ik]))
        ax.set_xticks(ind + width / 2)
        #ax.set_xticklabels(missCase)
        ax.set_xticklabels(missCase,rotation=60)
        if ik==0:
            ax.set_ylabel('RMSE',weight='bold')
            ax.legend((rects1[0], rects2[0]), ('DirectDist', 'BarrierDist'), loc=2)
        ymax = 1.6
        ax.set_ylim([0,ymax])
    #    if ik==1:
    #        ax.set_xlabel('#{:d} {:s}'.format(iStation,stations[iStation]),weight='bold')
    plt.suptitle('#{:d} {:s}'.format(iStation,stations[iStation]),weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/interp_samescale/sin_{:02d}.png'.format(iStation), dpi=150)
    plt.close(fig)
    
#%% All stations
ind = np.arange(nstation)
width = 0.2
#iCase = 0

for iCase in range(3):
    print('Printing:',iCase)
    fig, axes = plt.subplots(len(K),1, sharex=True, figsize=(7,5))
    for ik,ax in enumerate(axes):
        rmse_ID = ID['rmse'][:,ik,iCase]
        rmse_IB = IB['rmse'][:,ik,iCase]
        
        rects1 = ax.bar(ind, rmse_ID, width, color='grey')
        rects2 = ax.bar(ind+width, rmse_IB, width, color='k')
        
        ax.set_ylabel('k={:d}'.format(K[ik]),weight='bold')
        ax.set_xticks(ind + width / 2)
        #ax.set_xticklabels(missCase)
        iStationStr = ['#'+str(v) for v in range(nstation)]
        ax.set_xticklabels(iStationStr,rotation=60)
        if ik==0:
            ax.legend((rects1[0], rects2[0]), ('DirectDist', 'BarrierDist'), loc=1)
#        ymax = np.nanmax([ID['rmse'][:,:,iCase],IB['rmse'][:,:,iCase]])
        ymax = 1.1
        ax.set_ylim([0,ymax*1.1])
        ax.grid()
    plt.suptitle('RMSE   ({:s})'.format(missCase[iCase]),weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/interp/all_{:s}.png'.format(missCaseStr[iCase]), dpi=150)
    plt.close(fig)

#%% All stations
ind = np.arange(nstation)
width = 0.2
#iCase = 0

for iCase in range(3):
    print('Printing:',iCase)
    fig, axes = plt.subplots(len(K),1, sharex=True, figsize=(7,5))
    for ik,ax in enumerate(axes):
        rmse_ID = ID['rmse'][:,ik,iCase]
        rmse_IB = IB['rmse'][:,ik,iCase]
        
        rects1 = ax.bar(ind, rmse_ID, width, color='grey')
        rects2 = ax.bar(ind+width, rmse_IB, width, color='k')
        
        ax.set_ylabel('k={:d}'.format(K[ik]),weight='bold')
        ax.set_xticks(ind + width / 2)
        #ax.set_xticklabels(missCase)
        iStationStr = ['#'+str(v) for v in range(nstation)]
        ax.set_xticklabels(iStationStr,rotation=60)
        if ik==0:
            ax.legend((rects1[0], rects2[0]), ('DirectDist', 'BarrierDist'), loc=1)
#        ymax = np.nanmax([ID['rmse'][:,:,iCase],IB['rmse'][:,:,iCase]])
        ymax = 1.1
        ax.set_ylim([0,ymax*1.1])
        ax.grid()
    plt.suptitle('RMSE   ({:s})'.format(missCase[iCase]),weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/interp/all_{:s}.png'.format(missCaseStr[iCase]), dpi=150)
    plt.close(fig)

#%% Shelve
my_shelf = shelve.open('shelf/interp','n') # 'n' for new

shelveitems = ['IB','ID','SR','stations','nstation','K',
               'missCase','missCaseStr']

for key in shelveitems:
    try:
        my_shelf[key] = locals()[key]
    except:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

#%%
#shelvename = '/mnt/cstor01/home/lyin1/subgrid/shelves/'+loc_name
#shelvetmp = dir_shelve+loc_name
#shutil.copyfile(shelvename+'.dat', shelvetmp+'.dat')
#shutil.copyfile(shelvename+'.bak', shelvetmp+'.bak')
#shutil.copyfile(shelvename+'.dir', shelvetmp+'.dir')
#my_shelf = shelve.open(shelvetmp)
#for key in my_shelf:
#    try:
#        globals()[key]=my_shelf[key]
#    except:
#        pass
#my_shelf.close()

print("Total time: {:0.3f} sec".format(time.time() - start_time_total))
#sys.stdout = orig_stdout
#f.close()
