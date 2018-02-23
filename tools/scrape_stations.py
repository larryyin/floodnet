#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:44:31 2018

@author: larry
"""
#%%
from parse_stations import get_stations
from download_stations import download_station

#%%
S = get_stations()

#%%
import time
start_time = time.time()

for i in range(len(S['station'])):
    download_station(S['station'][i].replace(' ','_'),S['url'][i])
#download_station(station,url,Y=2011,M=7,D=1)

print("Job finished: {:0.3f} sec".format(time.time() - start_time))
