#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:00:50 2018

@author: larry
"""
import time
start_time = time.time()

import os
import pandas as pd
#%%
dir0 = 'series/obs/'
files = sorted(os.listdir(dir0))

file = files[0]
T = pd.read_csv(dir0+file,header=None,index_col=0,parse_dates=True,
                names=['t',file[:-4]])

for file in files[1:]:
    T = T.join(pd.read_csv(dir0+file,header=None,index_col=0,parse_dates=True,
                           names=['t',file[:-4]]))

T.to_csv('series/obs.csv')

#%%
dir0 = 'series/tid/'
files = sorted(os.listdir(dir0))

file = files[0]
T = pd.read_csv(dir0+file,header=None,index_col=0,parse_dates=True,
                names=['t',file[:-4]])

for file in files[1:]:
    T = T.join(pd.read_csv(dir0+file,header=None,index_col=0,parse_dates=True,
                           names=['t',file[:-4]]))

T.to_csv('series/tid.csv')

#%%
dir0 = 'series/mod/'
files = sorted(os.listdir(dir0))

file = files[0]
T = pd.read_csv(dir0+file,header=None,index_col=0,parse_dates=True,
                names=['t',file[:-4]])

for file in files[1:]:
    T = T.join(pd.read_csv(dir0+file,header=None,index_col=0,parse_dates=True,
                           names=['t',file[:-4]]))

T.to_csv('series/mod.csv')

#%%
print("Job finished: {:0.3f} sec".format(time.time() - start_time))
