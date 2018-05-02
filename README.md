# Floodnet
A deep neural network modeling framework to predict water levels based on regional observation data and hydrodynamic models - dealing with the spatial-temporal, cyclic characteristics and sparsity.  

## Introduction
Floodnet is a deep neural network architecture that captures all the available predictive potentials within a region to make the best water level prediction. Such predictive potentials for a typical inhabited coastal area are the harmonic tide and the past water levels recorded by one or multiple observation stations. For some areas where operational forecasting physical ocean models exist, the model results are also taken as predictive potentials. Non-water level types of data - wind, air pressure, temperature and salinity for examples – are not explored in this study due to their various levels of availability in different regions; we assume that their predictive values are at least partially embedded in the water levels provided by recent observations and hydrodynamic models.   

In specific, Floodnet assimilates the predictive values from 
1) harmonic tide,  
2) historical water level and surge at a single observation station,  
3) historical water level and surge at multiple observation stations and their spatial relationships,  
4) hindcast and forecast water level computed by hydrodynamic models at the observation stations and residues from the observations , and  
5) hindcast and forecast water level surfaces computed by hydrodynamic models.

# Designated Inverse Dropout (DID) method
A technique that handles missing data in neural network input.  

# Data
NYHOPS scraping [toolkit](https://github.com/larryyin/floodnet/raw/master/tools/scrapeNYHOPS.tar.gz "NYHOPS scraping toolkit")  
Resampled [hourly data](https://github.com/larryyin/floodnet/tree/master/data "NYHOPS hourly data")  

![Observation stations map](https://github.com/larryyin/floodnet/blob/master/img/02_map_obs_stations.png "Observation stations map")

![Available and selected observation stations](https://github.com/larryyin/floodnet/blob/master/img/00b_available_selected_obs.png "Available and selected observation stations")

# Single station predictability preliminary results
### Prediction random examples:
![Single station](https://github.com/larryyin/floodnet/blob/master/tests/24_6_obs_The_Battery_NY/check.png "Single station")  

![Single station](https://github.com/larryyin/floodnet/blob/master/tests/72_24_sur_tidall_Bergen_Point_West_Reach_NY/check.png "Single station")

### Compare the predictibility of different feature combinations:
![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_24_6_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_48_12_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_72_24_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_96_24_allstations.png "Single station")

### Compare the predictibility of different look-back periods:
![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_6_allx_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_12_allx_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_18_allx_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_24_allx_allstations.png "Single station")

### Compare the accuracy of different output lengths:
![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_24_ally_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_48_ally_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_72_ally_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_96_ally_allstations.png "Single station")

# Multi-station predictability preliminary results
### Prediction random examples:  
![Multi-station](https://github.com/larryyin/floodnet/blob/master/tests/multi/24_6_sur_tidall_Kings_Point_NY/check.png "Multi-station") 

![Multi-station](https://github.com/larryyin/floodnet/blob/master/tests/multi/72_24_sur_tidall_Kings_Point_NY/check.png "Multi-station")  
### Compare the single and multi-station predictions averaged over the 30 stations:  
![Multi-station](https://github.com/larryyin/floodnet/blob/master/img/sm_compare/rmse_compare_24_6_allstations.png "Multi-station")  

![Multi-station](https://github.com/larryyin/floodnet/blob/master/img/sm_compare/rmse_compare_72_24_allstations.png "Multi-station")  

# Spatial interpolation

### Least Distance Maps
![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/station_rank.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/leastdistmaps.png "Spatial interpolation")

### Interpolation comparison
![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/all_all.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/all_m0.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/all_m0p.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_00.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_01.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_02.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_03.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_04.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_05.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_06.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_07.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_08.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_09.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_10.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_11.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_12.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_13.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_14.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_15.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_16.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_17.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_18.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_19.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_20.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_21.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_22.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_23.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_24.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_25.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_26.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_27.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_28.png "Spatial interpolation")

![Spatial interpolation](https://github.com/larryyin/floodnet/blob/master/img/interp/sin_29.png "Spatial interpolation")

