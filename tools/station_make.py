#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 04:02:18 2018

@author: lyin1
"""
import numpy as np
import pandas as pd

from parse_stations import get_stations
S = get_stations()
stations = [v.replace(' ','_') for v in S['station']]
#select_id = [stations.index(v) for v in select]

lon = [float(v) for v in S['lon']]
lat = [float(v) for v in S['lat']]
#lat_max = max(lat)
#lat_min = min(lat)
#lon_max = max(lon)
#lon_min = min(lon)
#lat_diff = lat_max-lat_min
#lon_diff = lon_max-lon_min
#margin = .05

O = pd.DataFrame({'lon':lon,'lat':lat}, index=stations)

O.to_csv('stations_all.csv')

selection = np.load('stations_selected.npy')

OS = O.loc[selection]

OS.to_csv('stations_selected.csv')

lat_max = OS.lat.max()
lat_min = OS.lat.min()
lon_max = OS.lon.max()
lon_min = OS.lon.min()
lat_diff = lat_max-lat_min
lon_diff = lon_max-lon_min
#margin = .05

print(lat_max)
print(lat_min)
print(lon_max)
print(lon_min)
print(lat_diff)
print(lon_diff)