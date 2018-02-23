#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:06:06 2018

@author: larry
"""

import time
start_time = time.time()
#%%
import os
from parse_stations import get_stations
#from download_stations import download_period2
import requests
#import shutil
#%%
def download_period(station,url,beg,end,log):
    filename = 'data2/{:s}/{:s}_{:s}.csv'.format(station,beg,end)
    url = url.replace('$BEG$',beg)
    url = url.replace('$END$',end)
    try:
        r = requests.get(url)
        with open(filename, 'wb') as f:  
            f.write(r.content)
            str0 = '{:s}: {:s}_{:s} downloaded...\n'.format(station,beg,end)
            print(str0)
            log.write(str0)
        with open(filename,'r') as f:
            data_len = len(f.readlines()) # empty has 2 lines
        if data_len<=2:
            os.remove(filename)
            str1 = 'REMOVED {:s}: {:s}_{:s}...\n'.format(station, beg,end)
            print(str1)
            log.write(str1)
        return data_len
    except:
        return 0
#%%
S = get_stations()

records = os.listdir('log_error/')
#%%
stations = [v.strip('.log').replace('_',' ') for v in records]
ids = [S['station'].index(v) for v in stations]

#%%
for i,record in enumerate(records):
    with open('log_error/'+record,'r') as f:
        periods = [v.strip('\n').split(' to ') for v in f.readlines()]
        ymds = [[list(map(int,v.split('-'))) for v in vv] for vv in periods]
    
    log = open('log_download2/{:s}.log'.format(stations[i].replace(' ','_')),'w')
    try:
        os.makedirs('data2/'+stations[i].replace(' ','_'))
    except:
        pass
    #%%
    for error in ymds:
        if error[0][0]<error[1][0]:
            begs = ['{:4d}-{:02d}-01'.format(error[0][0],v) for v in range(7,13)]
            ends = list(begs)
            ends.pop(0)
            ends.append('{:4d}-{:02d}-01'.format(error[1][0],error[1][1]))
        elif error[0][0]==error[1][0]:
            begs = ['{:4d}-{:02d}-01'.format(error[0][0],v) for v in range(1,7)]
            ends = list(begs)
            ends.pop(0)
            ends.append('{:4d}-{:02d}-01'.format(error[1][0],error[1][1]))
#        print(begs,ends)
    #%%
        for beg,end in zip(begs,ends):
            data_len = download_period(S['station'][ids[i]].replace(' ','_'),
                                       S['url'][ids[i]],beg,end,log)
            if data_len<=2:
                with open('log_error2/{:s}.log'.format(stations[i].replace(' ','_')),'a') as log_err:
                        log_err.write('{:s} to {:s}\n'.format(beg,end))
    
    #%%
    log.close()

print("Job finished: {:0.3f} sec".format(time.time() - start_time))

#%%

