#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 01:52:12 2018

@author: lyin1
"""
import time
time_beg = time.time()
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import pandas as pd

npz = np.load('distm_data.npz')
SR = pd.read_csv('stations_distbar.csv')

lonm = npz['lonm']
latm = npz['latm']
dist_map = npz['dist_map']

lonmin = np.nanmin(lonm)
lonmax = np.nanmax(lonm)
latmin = np.nanmin(latm)
latmax = np.nanmax(latm)

#%%
for i in range(len(SR)):
#for i in range(7,30):
    #i = 0
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.pcolor(lonm,latm,dist_map[i],cmap='binary');
    ax.plot(SR.lon[i],SR.lat[i],'r.',markersize=5);
    ax.set_xlim([lonmin,lonmax]);
    ax.set_ylim([latmin,latmax]);
    #    ax.set_title(SR.station[i]);
    ax.text(lonmin,latmax,SR.station[i],ha='left',va='top')
    
    cbar = fig.colorbar(im)
    #cbar = grid.cbar_axes[-1].colorbar(im)
    
    #fig.suptitle(SR.station[i],fontweight='bold')
    plt.tight_layout()
    fig.savefig('leastdistmaps/{:02d}_{:s}.png'.format(i,SR.station[i]), bbox_inches='tight')
    plt.close(fig)
    print(i,'{:.6f} s'.format(time.time()-time_beg))


#%%
#fig = plt.figure(figsize=(11, 8.5))
#
#grid = AxesGrid(fig, 111,
#                nrows_ncols=(5, 6),
#                axes_pad=0.05,
#                cbar_mode='single',
#                cbar_location='right',
#                cbar_pad=0.1
#                )
#
#for i,ax in enumerate(grid):
#    print(i,time.time())
##    if i in [0,1,7]:
#    ax.set_axis_off()
#    im = ax.pcolor(lonm,latm,dist_map[i],cmap='binary');
#    ax.plot(SR.lon[i],SR.lat[i],'r.',markersize=5);
#    ax.set_xlim([lonmin,lonmax]);
#    ax.set_ylim([latmin,latmax]);
##    ax.set_title(SR.station[i]);
#    ax.text(lonmin,latmax,SR.station[i],fontsize=6,ha='left',va='top')
#
#cbar = ax.cax.colorbar(im)
#cbar = grid.cbar_axes[-1].colorbar(im)
#
#fig.suptitle('Least Distance Maps (x100m)',fontweight='bold')
#fig.savefig('least_dist_maps.png', bbox_inches='tight',dpi=300)




print('{:.6f} s'.format(time.time()-time_beg))