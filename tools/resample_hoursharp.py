#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 20:49:22 2018

@author: lyin1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:40:25 2018
@author: larry
"""
import time
start_time = time.time()

import numpy as np
import pandas as pd

#%%
otm = ['obs','tid','mod']
for v in otm:
    print(v)
    T = pd.read_csv('series/'+v+'.csv',header=0,index_col=0,parse_dates=True)
    for i in range(len(T)):
        if T.index[i].minute==0:
            break
    Th = T.resample('H').mean()
    hrange = pd.date_range(start=T.index[i],end=T.index[-1],freq='H')
    idx = np.intersect1d(T.index,hrange)
    Th = T.loc[idx]
    Th.to_csv('series/'+v+'_hs.csv')

#%%
print("Job finished: {:0.3f} sec".format(time.time() - start_time))