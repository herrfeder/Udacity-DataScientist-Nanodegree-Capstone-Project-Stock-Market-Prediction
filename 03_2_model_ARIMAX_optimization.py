# 03_2 Testing on all Parameters for finding best SARIMAX Model doing Cross Validation

import data_prep_helper
import statsmodels.api as sm
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import os


do = data_prep_helper.ShiftChartData(chart_col=["Price", "High", "Low"])

df = data_prep_helper.ShiftChartData.get_causal_const_shift(do.chart_df,)


# this for loop will control the the features to include for combination and testing
ext_cols = []
for col in df.columns:
    if (col.endswith('month')) :
        ext_cols.append(col)
        
def feature_comb_iter(i, ext_cols):
    '''
    This function will take a list of features and will run them against a Multivariate SARIMAX Model Fitting.
    It will do multiple Splits using my own helper class for getting more robust cross validated results.
    All results will be stored into a single csv for a number of features.
    
    INPUT:
        i - (int) Number of features to test on
        
    OUTPUT:
        csv - (csv) File that holds all cross validation results with Features, Correlation and RMSE
    '''
    result_list = []
    
    # will produce an iterator with all different combinations
    for comb in itertools.combinations(ext_cols, i):
        result_dict = {}
        split_index = 0
        
        # iterator for multiple splits
        for train, test in do.gen_return_splits():
            split_index = split_index + 1
            exog_s1i1 = train[list(comb)]
            exog = test[list(comb)]
            s1i1 = train['bitcoin_Price']
            arimax = sm.tsa.statespace.SARIMAX(s1i1, exog=exog_s1i1,
                                       enforce_invertibility=False, 
                                       enforce_stationarity=False, 
                                       freq='D').fit(disp=0)

            forecast = arimax.get_forecast(steps=len(test), exog=exog)

            result_dict["split_{}_CORR".format(split_index)] = np.corrcoef(forecast.predicted_mean,test["bitcoin_Price"].values)[0][1]
            result_dict["S_{}_RMSE".format(split_index)] = sqrt(mean_squared_error(forecast.predicted_mean, test["bitcoin_Price"]))
    
        result_dict["FEATURES"] = str(comb)
        result_list.append(result_dict)
        
    
    result_df = pd.DataFrame(result_list)
    result_df["NUM_FEATURES"] = i   
    result_df.to_csv("arimax_results/arimax_split_combinations_results_{}.csv".format(i))
    
    
if __name__ == '__main__':
    ## Doing a multiprocessing approach seems to slow down the whole process dramatically
    #pool = Pool(4)           # Create a multiprocessing Pool
    #pool.map(feature_comb_iter, range(4,10))  # process range
    feature_comb_iter(8, ext_cols)