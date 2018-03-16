#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 02:13:24 2018

@author: larry
"""
import time
time_beg = time.time()
import numpy as np
from subprocess import call

STATION = np.load('stations_selected.npy')
FEATURE = ['obs','sur','obs_tid','obs_tidext','sur_tidext']
XYL = [(24,6,[20,10]),
       (24,12,[20,10]),
       (24,18,[20,10]),
       (24,24,[20,10]),
       (48,6,[40,20]),
       (48,12,[40,20]),
       (48,18,[40,20]),
       (48,24,[40,20]),
       (72,6,[64,32]),
       (72,12,[64,32]),
       (72,18,[64,32]),
       (72,24,[64,32]),
       (96,6,[100,50]),
       (96,12,[100,50]),
       (96,18,[100,50]),
       (96,24,[100,50])
      ]
#XL = [24,48,72,96,120,192,240,360,720]
#YL = [6,12,18,24,48,72,96]
#LAYERS

#call(['python','single.py','--feature',FEATURE[3],'--xl',str(48),'--yl',str(12),
#      '--station',STATION[0],'--layers',str(30),str(10)])
count = 0
for x_len,y_len,layers in XYL:
    for station in STATION:
        for feature in FEATURE:
            count += 1
            print('{:d}: {:d}_{:d}_{:s}_{:s}_{:s}...'.format(count,x_len,y_len,feature,str(layers),station))
#            python single.py --feature sur_tidext --xl 48 --yl 12 --station Rockaway_Inlet_near_Floyd_Bennett_Field_NY --layers 30 10
            time_sub_beg = time.time()
            call(['python','single.py','--feature',feature,
                  '--xl',str(x_len),'--yl',str(y_len),'--station',station,
                  '--layers',str(layers[0]),str(layers[1])])
            time_sub_end = time.time()
            print("time: {:.0f}s, total: {:.0f}m".format(time_sub_end-time_sub_beg,(time_sub_end-time_beg)/60))