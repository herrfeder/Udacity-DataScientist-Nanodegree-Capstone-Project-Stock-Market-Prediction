# Capstone Project: Multivariate Timeseries Analysis and Prediction (Stock Market)
## Purpose

Building an Time Series Forecast Application to predict and forecast __Bitcoin financial data__
using supervised and unsupervised Machine Learning Approaches, this includes:
  * search, collection and of supportive Features in form of suitable Time Series (social media, other similar charts)
  * preparation, analysis, merging of Data and Feature Engineering using:
    * Correlative Analysis
    * Stationarity Analysis
    * Causality Analysis
  * Model Preprocessing and Model Fitting with this Machine Learning Algorithms:
    * supervised SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model)
    * unsupervised GRU (Gated Recurrent Unit)
  * building an Web Application using a Dash Webapp (see folder __webapp__)
    * explains my roadmap of analysis and conclusions
    * provides feature of daily forecasting using designed models
    * Own Webapp Repository: https://github.com/herrfeder/Udacity-Data-Scientist-Capstone-Multivariate-Timeseries-Prediction-Webapp
  

## Approach/Idea

It's nearly impossible to give an accurate prediction for Stock Charts or Cryptocurrency Charts for the Future.
Therefore I will only try to find signals or triggers that may announce major Movements on the Bitcoin Chart and may occur
right before the real movements.

I want to find Correlation and Causality to the Bitcoin Price by shifting all other collected time series in time.
For Example: Shifting all supportive Features one month to past gives me the freedom to look one month into the future for forcasting.
These notebooks will show my course of action:

  * [01 Correlation Analysis](https://github.com/herrfeder/DataScientist/blob/master/Project_05_Capstone_Stock_Chart_Analysis/01_corr_analysis.ipynb)
  * [02 Stationarity and Causality Analysis](https://github.com/herrfeder/DataScientist/blob/master/Project_05_Capstone_Stock_Chart_Analysis/stationarity_causality_analysis.ipynb)
  * [03 SARIMA Modelling](https://github.com/herrfeder/DataScientist/blob/master/Project_05_Capstone_Stock_Chart_Analysis/03_1_model_ARIMAX.ipynb)
  * [04 GRU Modelling](https://github.com/herrfeder/DataScientist/blob/master/Project_05_Capstone_Stock_Chart_Analysis/04_model_GRU.ipynb)
  * [05 Decision Algorithm](https://github.com/herrfeder/DataScientist/blob/master/Project_05_Capstone_Stock_Chart_Analysis/05_decision_algorithm.ipynb)



## Screenshots of Webapp

| Forecast Application | Buy And Sell Simulation | Timeshift Correlation |
|--------------------------------------|--------------------------------------|--------------------------------------|
| ![](https://github.com/herrfeder/DataScientist/raw/master/Project_05_Capstone_Stock_Chart_Analysis/images/forecast_full_view.png) | ![](https://github.com/herrfeder/DataScientist/raw/master/Project_05_Capstone_Stock_Chart_Analysis/images/buy_and_sell_sim.png) | ![](https://github.com/herrfeder/DataScientist/raw/master/Project_05_Capstone_Stock_Chart_Analysis/images/corr_timeshift.png) |


## Used Data

1. Stock Market Data for the last five years from [Investing.com](https://www.investing.com) for:
  * Bitcoin, DAX, SP500, Google, Amazon, Alibaba
2. Google Trends for keywords "bitcoin", "cryptocurrency", "ethereum", "trading", "etf" using this notebook 
  * [00_scrape_googletrend.ipynb](https://github.com/herrfeder/DataScientist/blob/master/Project_05_Capstone_Stock_Chart_Analysis/00_scrape_googletrend.ipynb)
3. Twitter Sentiments for keyword "bitcoin" and "#economy" using notebooks 
  * [00_scrape_twitter.py](blubb)
  * [00_tweet_to_sent.ipynb](blubb)

## Used Libaries

  * Data Collection:
    * [twint](https://github.com/twintproject/twint)
    * [pytrends](https://github.com/GeneralMills/pytrends)
  * NLP:
    * [NLTK](https://github.com/nltk/nltk)
  * Webapp and Visualisation: 
    * [Plotly](https://github.com/plotly/plotly.py)
    * [dash](https://github.com/plotly/dash)
    * [matplotlib](https://github.com/matplotlib/matplotlib)
  * Data Analysis and Wrangling:
    * [Pandas](https://github.com/pandas-dev/pandas)
    * [Numpy](https://github.com/numpy/numpy)
    * [statsmodels](https://github.com/statsmodels/statsmodels)
  * Modelling and Evaluation:
    * [Numpy](https://github.com/numpy/numpy)
    * [Scikit Learn](https://github.com/scikit-learn/scikit-learn)
    * [statsmodels](https://github.com/statsmodels/statsmodels)
    * [Tensorflow](https://github.com/tensorflow/tensorflow)
    * [Keras](https://github.com/keras-team/keras)



## Included Other Files
  
  * __webapp__: Folder that holds the files and folders for the Dash webapp. For installation and deployment, look into it
  * __00_scrape_googletrend.ipynb__: Scraping Google Trends
  * __00_scrape_twitter.py__: Scrape Twitter using Twint
  * __00_tweet_to_sent.ipynb__: Convert collected tweets to sentiment scores
  * __01_corr_analysis.ipynb__: Data Processing, Merging and Correlative Analysis
  * __02_stationarity_causality_analysis.ipynb__: Analysis for Stationarity and Causality
  * __03_1_model_ARIMAX__: Modelling and Validation for SARIMAX model
  * __03_2_model_ARIMAX_optimization.py__: Optimizing SARIMAX model by testing different sets of features
  * __03_3_investigate_feature_optimization.ipynb__: Finding model with best performance from previous test
  * __04_model_GRU.ipynb__: Modelling and Validation for GRU model
  * __data/__: Holds all source data described as above
  * __arimax_results/__: holds the results for SARIMAX feature optimization
  * __data_prep_helper.py__: consists of helper classes to read, process, shift and split data and do forecasting
  * __plot_helper.py__: consists of different supportive plotly functions 

## Brief Results

The model prediction for using time series that aren't shifted far into the past like up to a week are pretty accurate.
The model prediction for the desired month is far away from beeing accurate but we can see several volatile Chart Movements in forms of signals and triggers before they will happen and that's a nice result. It seems by averaging the forecast we can get good recommendations for Buy and Sell decisions.

### Possible Roadmap/Chances

  * Extensive Hyperparameter Optimization: Due to a lack of time, resources and knowledge this was only done rudimentary. I'm sure the models can be improved
    by that.
  * Extend Webapp to full realtime Forecasting.
  * Check more and more different feature time series.
  * Get a better understanding of Deep Learning RNN's like GRU


## Installation and Deployment

I prepared a Dockerfile that should automate the installation and deployment. 
For further instructions see folder [webapp](https://github.com/herrfeder/Udacity-Data-Scientist-Capstone-Multivariate-Timeseries-Prediction-Webapp)

I'm running the Web Application temporary on my own server: https://federland.dnshome.de/bitcoinprediction
Please be gentle, it's running on limited resources. This app __isn't responsive__.

## Acknowledgements

  * To [Nicolas Essipova](https://github.com/NicoEssi) for being my mentor for the whole Nanodegree
  * To all the authors of explanations, code snippets and functions I mention in Resources
  
## Resources

### Causality Resources

  * General Explanation and Tutorial for Stationarity Analysis of Time Series:
    * https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
  * Explanation of Difference Correlation and Causalisation: 
    * https://calculatedcontent.com/2013/05/27/causation-vs-correlation-granger-causality/
  * Used Granger Causality Function from: 
    * https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

### SARIMAX Resources

  * Good Overview about ARIMA Models: 
    * https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
  * How to prepare Explanatory Variables for Multivariate SARIMAX Model: 
    * https://www.kaggle.com/viridisquotient/arimax
  * How to prepare Time Series data for Multivariate SARIMAX Model: 
    * https://www.machinelearningplus.com/time-series/time-series-analysis-python/

### GRU Resources

  * Great visual explanation of RNN (LSTM/GRU):
    * https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
  * Data Preprocessing for Keras GRU:
    * https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras
  * The GRU Code is inspired and Model Concept is used from there: 
    * https://github.com/ninja3697/Stocks-Price-Prediction-using-Multivariate-Analysis/blob/master/Multivatiate-GRU/Multivariate-3-GRU.ipynb
