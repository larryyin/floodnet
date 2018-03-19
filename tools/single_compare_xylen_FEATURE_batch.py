#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:17:57 2018

@author: larry
"""
import time
time_beg = time.time()
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def selectDIR(path='tests/',xylen=None,feature=None,station=None):
    if xylen==None:
        xylen = '*'
    if feature==None:
        feature = '*'
    if station==None:
        station = '*'
    testname = '{:s}*{:s}*{:s}'.format(xylen,feature,station)
    if feature=='obs':
        DIR = list(set(glob.glob(path+testname))-set(glob.glob(path+testname.replace(feature,'obs_tid'))))
    if feature=='sur':
        DIR = list(set(glob.glob(path+testname))-set(glob.glob(path+testname.replace(feature,'sur_tid'))))
    if feature=='obs_tid':
        DIR = list(set(glob.glob(path+testname))-set(glob.glob(path+testname.replace(feature,'obs_tidall'))))
    if feature=='obs_tidall':
        DIR = glob.glob(path+testname)
    if feature=='sur_tidext':
        DIR = glob.glob(path+testname)
    if feature=='sur_tidall':
        DIR = glob.glob(path+testname)
    return DIR

#%%
STATION = np.load('stations_selected.npy')
dir0 = 'img/'
dir1 = 'tests/'
dir2 = 'rmse_compare_xylen_FEATURE/'
FEATURE = ['obs','sur','obs_tid','obs_tidall','sur_tidext','sur_tidall']

#xylen = '72_24'
#XYLEN = ['24_12','24_18','24_24',
#         '48_6','48_12','48_18','48_24',
#         '72_6','72_12','72_18','72_24',
#         '96_6','96_12','96_18','96_24']
#XYLEN = ['72_12','72_18','72_24',
#         '96_6','96_12','96_18','96_24']
XYLEN = ['72_6']

for xylen in XYLEN:
    print('Processing',xylen)
    #testname = '24_6_{:s}*'.format(FEATURE[0])
    #DIR = glob.glob('tests/'+testname)
    
    #%%
    RMSE_TEST_TS = []
    for feature in FEATURE:
        DIR = selectDIR(path=dir1,xylen=xylen,feature=feature)
    #    example = np.load(DIR[0]+'/runinfo.npz')
        #station = str(example['station'])
    #    feature = str(example['feature'])
    #    x_len = int(example['x_len'])
    #    y_len = int(example['y_len'])
        RMSE_TEST_TS_sub = []
        for v in DIR:
            run = np.load(v+'/runinfo.npz')
            RMSE_TEST_TS_sub.append(run['rmse_test_ts'])
        RMSE_TEST_TS.append(np.array(RMSE_TEST_TS_sub).mean(axis=0))
    TESTNAME = FEATURE
    
    example = np.load(DIR[0]+'/runinfo.npz')
    #station = str(example['station'])
    #feature = str(example['feature'])
    x_len = int(example['x_len'])
    y_len = int(example['y_len'])
    #%%
    dict = {v0:v1 for v0,v1 in zip(TESTNAME,RMSE_TEST_TS)}
    P = pd.DataFrame(dict)[FEATURE]
    
    #%%
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    P.plot(ax=ax)
    ax.set_ylabel('rmse (m)')
    ax.set_xlabel('timestep')
    ax.set_title('X:{:d}h   Y:{:d}h   All stations'.format(x_len,y_len),weight='bold')
    fig.tight_layout()
    plotname = 'rmse_compare_{:d}_{:d}_allstations.png'.format(x_len,y_len)
    fig.savefig(dir0+dir2+plotname, format='png', dpi=300)
    plt.close(fig)
#%%
print('{:.4f}s'.format(time.time()-time_beg))