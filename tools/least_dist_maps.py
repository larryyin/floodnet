#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 02:19:03 2018

@author: lyin1
"""
import time
time_beg = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skfmm

def nearest_ixy(x,y,xgrid,ygrid):
    diff = (xgrid-x)**2+(ygrid-y)**2
    diff_min = np.nanmin(diff)
    [iy,ix] = np.nonzero(diff==diff_min)
    return int(ix),int(iy)

def dist_greatcircle(lat1,lon1,lat2,lon2):
    R = 6371000   # m
    latrad1 = np.deg2rad(lat1)
    latrad2 = np.deg2rad(lat2)
    dLat = latrad2-latrad1
    dLon = np.deg2rad(lon2-lon1)
    a = (np.sin(dLat/2) * np.sin(dLat/2) +
        np.cos(latrad1) * np.cos(latrad2) *
        np.sin(dLon/2) * np.sin(dLon/2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


D = pd.read_csv('water_mask.csv',usecols=['I','J','lon','lat','water'])

wm = np.zeros([D.J.max(),D.I.max()])
lonm = np.zeros([D.J.max(),D.I.max()])
latm = np.zeros([D.J.max(),D.I.max()])
Im = np.zeros([D.J.max(),D.I.max()])
Jm = np.zeros([D.J.max(),D.I.max()])

for v in D.values:
    I = v[0].astype(int)
    J = v[1].astype(int)
    w = v[4].astype(int)
    Im[J-1,I-1] = I
    Jm[J-1,I-1] = J
    lonm[J-1,I-1] = v[3]
    latm[J-1,I-1] = v[2]
    wm[J-1,I-1] = w

print('Processing hydraulic connectivity...')

current_output, num_ids = ndimage.label(wm)

# Plot outputs
#plt.imshow(data_bool, cmap="spectral", interpolation='nearest')
#plt.imshow(current_output, cmap="spectral", interpolation='nearest')

zone_id = range(num_ids+1)
zone_count = []

for i in zone_id:
    zone = current_output==i
    zone_count.append(np.sum(zone))

inun_id = np.array(zone_count).argsort()[-2]
inun_zone = current_output==inun_id
inun = np.zeros(wm.shape)
inun[inun_zone] = 1
wcm = inun.astype(bool)

#%%
cnJ,cnI=latm.shape
s = []
s += "I,J,lat,lon,water"
for j in range(cnJ):
    for i in range(cnI):
        s += "\n{I:.0f},{J:.0f},{lat:.6f},{lon:.6f},{water:.0f}".format(I=Im[j][i],J=Jm[j][i],lat=latm[j][i],lon=lonm[j][i],water=inun[j][i]) 
with open('water_connected.csv', 'w') as f:
    f.writelines(s)

#%%
S = pd.read_csv('stations_selected.csv',index_col=0)

# IX, IY based on 0
lonm_wcm = np.copy(lonm)
latm_wcm = np.copy(latm)
lonm_wcm[~wcm] = np.nan
latm_wcm[~wcm] = np.nan
IX = []
IY = []
for i in range(len(S)):
    ix,iy = nearest_ixy(S.lon[i],S.lat[i],lonm_wcm,latm_wcm)
    IX.append(ix)
    IY.append(iy)

S['IX'] = IX
S['IY'] = IY

#%%
dist_map = []
for i in range(len(S)):
    dist_grid = np.ones(wcm.shape,dtype='int')
    dist_grid[S.IY[i],S.IX[i]] = 0
    dist_case = np.ma.masked_array(dist_grid, ~wcm)
    dist_map.append(skfmm.distance(dist_case))

#%%
distm = np.array([v[IY,IX].data for v in dist_map])
DB = pd.DataFrame(distm,columns=S.index,index=S.index)
DB.to_csv('distbar_matrix.csv')

SR = pd.DataFrame({'std':DB.std(),'id0':range(len(DB))}).sort_values('std')
SR['id'] = range(len(SR))
SR['station'] = SR.index
SR.set_index('id',inplace=True)

#%% Order by dist_bar
SR = pd.concat([SR.set_index('station'),S.iloc[SR.id0,:]],axis=1)
SR['id'] = range(len(SR))
SR['station'] = SR.index
SR.set_index('id',inplace=True)

dist_map = [dist_map[i] for i in SR.id0.values]

distm = np.array([v[SR.IY,SR.IX].data for v in dist_map])
DB = pd.DataFrame(distm,columns=SR.index,index=SR.index)
DB.to_csv('distbar_matrix.csv')

SR.to_csv('stations_distbar.csv')

#%% GC dist
distgm = np.array([dist_greatcircle(v1,v2,SR.lat.values,SR.lon.values) for v1,v2 in zip(SR.lat.values,SR.lon.values)])

#%%
np.savez('distm_data',wcm=wcm,Im=Im,Jm=Jm,wm=wm,lonm=lonm,latm=latm,
         current_output=current_output,num_ids=num_ids,zone_count=zone_count,zone_id=zone_id,
         s_index=SR.index.values,s_col=SR.columns.values,s_values = SR.values,
         dist_map=dist_map,distm=distm,distgm=distgm)

print('{:.6f} s'.format(time.time()-time_beg))