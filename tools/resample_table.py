#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:40:25 2018

@author: larry
"""
import time
start_time = time.time()

import pandas as pd
#%%
otm = ['obs','tid','mod']
for v in otm:
    T = pd.read_csv('series/'+v+'.csv',header=0,index_col=0,parse_dates=True)
    Th = T.resample('H').mean()[T.resample('H').count()>3]
    Th.to_csv('series/'+v+'_h.csv')

#%%
print("Job finished: {:0.3f} sec".format(time.time() - start_time))