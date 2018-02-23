#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s = "123123STRINGabcabc"

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""


print find_between( s, "123", "abc" )
print find_between_r( s, "123", "abc" )

123STRING
STRINGabc
"""
#%%
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

#%%
def get_stations():
    with open('stations.txt','r') as f:
        raw = f.readlines()
        
    raw_clean = [v.replace('\t','').replace('\n','') for v in raw]
    
    #%%
    STATION = [find_between(v,"title: '","',  mid:") for v in raw_clean[0:-1:17]]
    DID = [find_between(v,", did: '","', mdiff:") for v in raw_clean[0:-1:17]]
    MID = [find_between(v,",  mid: '","', did:") for v in raw_clean[0:-1:17]]
    MDIFF = [find_between(v,", mdiff: ",", ") for v in raw_clean[0:-1:17]]
    DIFF = [find_between(v,"diff: ",", lat:") for v in raw_clean[1:-1:17]]
    LAT = [find_between(v,", lat: ",", lon") for v in raw_clean[1:-1:17]]
    LON = [find_between(v,", lon: ",", map_visible:") for v in raw_clean[1:-1:17]]
    NAVD88_OFF = [find_between(v,"navd88_offset: ", ",") for v in raw_clean[6:-1:17]]
    TIDE_DIFF = [find_between(v,", ext_tide_shift: ",",") for v in raw_clean[10:-1:17]]
    
    #%%
    URL = []
    for i in range(len(STATION)):
        url = 'http://hudson.dl.stevens-tech.edu/sfas/d/comma_sfas_query.php?&diff={:s}&type=ELEV&did={:s}&mid={:s}&units=metric&start=$BEG$&end=$END$&location={:s}&parameter=Water%20Level%20relative%20to%20&offset=NAVD88&off={:s}&time=GMT&mdiff={:s}&tide_diff={:s}&forecast=OPSE01&station_type=T&nothing=4'.format(
                DIFF[i],DID[i],MID[i],STATION[i].replace(' ','%20'),NAVD88_OFF[i],
                MDIFF[i],TIDE_DIFF[i])
        URL.append(url)
    
    return {'station':STATION,
            'lat':LAT,
            'lon':LON,
            'url':URL}
