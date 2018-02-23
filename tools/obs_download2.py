#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:20:17 2018

@author: larry
"""
#%%
import requests
#import shutil
import os

#%%
def download_period(station,url,beg,end,log):
    filename = '{:s}/{:s}_{:s}.csv'.format(station,beg,end)
    url = url.replace('$BEG$',beg)
    url = url.replace('$END$',end)
    try:
        r = requests.get(url)
        with open(filename, 'wb') as f:  
            f.write(r.content)
            str0 = '{:s} downloaded...\n'.format(filename)
            print(str0)
            log.write(str0)
        with open(filename,'r') as f:
            data_len = len(f.readlines()) # empty has 2 lines
        if data_len<=2:
            os.remove(filename)
            str1 = 'EMPTY FILE {:s} REMOVED...\n'.format(filename)
            print(str1)
            log.write(str1)
        return data_len
    except:
        return 0

def download_station(station,url,Y=2017,M=1,D=1):
#    if os.path.exists(station):
#        shutil.rmtree(station)
    try:
        os.makedirs(station)
    except:
        pass
    
    counter = 0
    empty_counter = 0
    data_len = 3
    ERROR = []
    log = open('log_download/{:s}.log'.format(station),'w')
#    while data_len>2:
    while empty_counter<=3:
        counter += 1
        print(counter)
        end = '{:4d}-{:02d}-{:02d}'.format(Y,M,D)
        if M==1:
            M = 7
            Y -= 1
        elif M==7:
            M = 1
        beg = '{:4d}-{:02d}-{:02d}'.format(Y,M,D)
        data_len = download_period(station,url,beg,end,log)
        if data_len<=2:
            empty_counter += 1
            ERROR.append('{:s} to {:s}\n'.format(beg,end))
    
    log.close()
    with open('log_error/{:s}.log'.format(station),'a') as log_err:
                log_err.writelines(ERROR[:-2])

#%%
station = 'battery'
url = 'http://hudson.dl.stevens-tech.edu/sfas/d/comma_sfas_query.php?&diff=-1.849&type=ELEV&did=N017&mid=M007t&units=metric&start=$BEG$&end=$END$&location=The%20Battery%20NY&parameter=Water%20Level%20relative%20to%20&offset=NAVD88&off=0&time=GMT&mdiff=0&tide_diff=-1.849&forecast=OPSE01&station_type=T&nothing=4'
#station = 'bergen_point'
#url = 'http://hudson.dl.stevens-tech.edu/sfas/d/comma_sfas_query.php?&diff=-2.191&type=ELEV&did=N019&mid=M008t&units=metric&start=$BEG$&end=$END$&location=Bergen%20Point%20West%20Reach%20NY&parameter=Water%20Level%20relative%20to%20&offset=NAVD88&off=0&time=GMT&mdiff=0&tide_diff=-2.191&forecast=OPSE01&station_type=T&nothing=4'
#S = get_stations()


#%%
import time
start_time = time.time()

download_station(station,url)
#download_station(station,url,Y=2011,M=7,D=1)

print("Job finished: {:0.3f} sec".format(time.time() - start_time))


#%%
