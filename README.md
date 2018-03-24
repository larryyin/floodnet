# FLOODNET
A deep neural network modeling framework to predict water levels based on regional observation data and hydrodynamic models - dealing with the spatial-temporal, cyclic characteristics and sparsity.  

## Data
Scraped NYHOPS with these [scripts](https://github.com/larryyin/floodnet/raw/master/tools/scrapeNYHOPS.tar.gz "NYHOPS scraping toolkit")  
Resampled hourly [data](https://github.com/larryyin/floodnet/tree/master/data "NYHOPS hourly data")  

![Observation stations map](https://github.com/larryyin/floodnet/blob/master/img/02_map_obs_stations.png "Observation stations map")

![Available and selected observation stations](https://github.com/larryyin/floodnet/blob/master/img/00b_available_selected_obs.png "Available and selected observation stations")

## Single station predictability preliminary results
Prediction random examples:
![Single station](https://github.com/larryyin/floodnet/blob/master/tests/24_6_obs_The_Battery_NY/check.png "Single station")  

![Single station](https://github.com/larryyin/floodnet/blob/master/tests/72_24_sur_tidall_Bergen_Point_West_Reach_NY/check.png "Single station")

Compare the predictibility of different feature combinations:
![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_24_6_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_48_12_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_72_24_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_xylen_FEATURE/rmse_compare_96_24_allstations.png "Single station")

Compare the predictibility of different look-back periods:
![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_6_allx_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_12_allx_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_18_allx_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_y_XLEN/rmse_compare_24_allx_allstations.png "Single station")

Compare the accuracy of different output lengths:
![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_24_ally_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_48_ally_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_72_ally_allstations.png "Single station")

![Single station](https://github.com/larryyin/floodnet/blob/master/img/rmse_compare_feature_x_YLEN/rmse_compare_96_ally_allstations.png "Single station")
