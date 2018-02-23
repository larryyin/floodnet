#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:51:15 2018

@author: larry
"""
import time
start_time = time.time()

import os
import pandas as pd
from parse_stations import get_stations

S = get_stations()

#%%
log0 = open('concat_nosuchfolder.log','w')
log1 = open('concat_errors.log','w')

#for d1 in sorted(os.listdir(d0)):
#for d1 in [sorted([v.replace(' ','_') for v in S['station']])[0]]:
for d1 in sorted([v.replace(' ','_') for v in S['station']]):
    OBS = pd.Series()
    TID = pd.Series()
    MOD = pd.Series()
    for d0 in ['data','data2']:
        try:
            for d2 in sorted(os.listdir('{:s}/{:s}/'.format(d0,d1))):
                d3 = '{:s}/{:s}/{:s}'.format(d0,d1,d2)
                try:
                    print(d3)
                    T = pd.read_csv(d3)
                    T['t'] = pd.to_datetime(T['DateTime (GMT)'])
                    T.drop(['DateTime (GMT)'], axis=1, inplace=True)
                    T.set_index('t', inplace=True)
                    try:
                        obs = T['Observed Water Level relative to  NAVD88 (m)'].dropna()
                        OBS = OBS.append(obs)
                    except:
                        pass
                    try:
                        tid = T['Astronomical Tide relative to NAVD88 (m)'].dropna()
                        TID = TID.append(tid)
                    except:
                        pass
                    try:
                        mod = T['OPSE01-predicted Water Level relative to  NAVD88 (m)'].dropna()
                        MOD = MOD.append(mod)
                    except:
                        pass
                except:
                    log1.write(d3+'\n')
        except:
            log0.write('{:s}/{:s}\n'.format(d0,d1))
    OBS = OBS[~OBS.index.duplicated(keep='first')].sort_index()
    TID = TID[~TID.index.duplicated(keep='first')].sort_index()
    MOD = MOD[~MOD.index.duplicated(keep='first')].sort_index()
    filename0 = 'series/obs/{:s}.csv'.format(d1)
    filename1 = 'series/tid/{:s}.csv'.format(d1)
    filename2 = 'series/mod/{:s}.csv'.format(d1)
    try:
        os.remove(filename0)
    except OSError:
        pass
    try:
        os.remove(filename1)
    except OSError:
        pass
    try:
        os.remove(filename2)
    except OSError:
        pass
    if len(OBS)>1:
        OBS.to_csv(filename0)
    if len(TID)>1:
        TID.to_csv(filename1)
    if len(MOD)>1:
        MOD.to_csv(filename2)

log0.close()
log1.close()

#%%
print("Job finished: {:0.3f} sec".format(time.time() - start_time))

