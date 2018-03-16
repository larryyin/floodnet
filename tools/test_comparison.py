#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:17:57 2018

@author: larry
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
STATION = np.load('stations_selected.npy')
testname = '24_6*{:s}*'.format(STATION[2])
DIR = glob.glob('tests/'+testname)
dir0 = 'img/'

example = np.load(DIR[0]+'/runinfo.npz')
station = str(example['station'])
x_len = int(example['x_len'])
y_len = int(example['y_len'])
#%%
#['EPOCH', 'MSE_TRAIN', 'MSE_DEV', 'mse_test_ts', 'mse_dev', 'mse_test']
RMSE_TEST_TS = []
TESTNAME = []
for v in DIR:
    run = np.load(v+'/runinfo.npz')
    RMSE_TEST_TS.append(run['rmse_test_ts'])
    TESTNAME.append(str(run['feature']))
RMSE_TEST_TS = np.array(RMSE_TEST_TS)

#%%
dict = {v0:v1 for v0,v1 in zip(TESTNAME,RMSE_TEST_TS)}
P = pd.DataFrame(dict)[['obs','sur','obs_tid','obs_tidext','sur_tidext']]

#%%
fig, ax = plt.subplots(1,1, figsize=(8, 4))
P.plot(ax=ax)
ax.set_ylabel('rmse (m)')
ax.set_xlabel('timestep')
ax.set_title('{:s}\nX:{:d}h   Y:{:d}h'.format(station,x_len,y_len),weight='bold')
fig.tight_layout()
plotname = 'rmse_compare_{:s}_{:d}_{:d}.png'.format(station,x_len,y_len)
fig.savefig(dir0+plotname, format='png', dpi=300)
plt.close(fig)