thermometr
==========

### Installation
Clone this repo. Change directories into thermometr and run ```python setup.py install``` to install the package thermometr 
### Overview
The Thermometr class implements the [Twitter S-H-ESD Algorithm](https://blog.twitter.com/2015/introducing-practical-and-robust-anomaly-detection-in-a-time-series)  with an ARIMA based validation. The repository is setup as an installable python package to be used elsewhere in the Product Analytics codebase.

#### Thermometr Class
The thermometer class takes in an iterable (or list/dict of iterables) or pandas dataframe of time series data. The constructor standardizes the input to a list of numpy arrays. The constructor also takes in an iterable of dates that can be used instead of numerical indices in results. 

The detect() method can be used to find all the anomalies in all the series that exist in the instance. The detect_latest() method only returns the anomalies that are the last value in each of the series. A number of private helper functions exist within the class with descriptive docstrings. 

#### S-H-ESD
This algorithm is an extension of the Extreme Studentized Deviate Test (Grubb's Test) for finding outliers. To find outliers, the timeseries is decomposed, using Seasonal-Trend Decomposition, (STL), into a timeseries of seasonality and growth trend. The difference from the observed data and the (seasonal + trend) is called the residual. The residual can be thought of as the distance between the observed points and the curve of best fit calculated by STL. The assumption that makes S-H-ESD work is that the residual is symmetrically distributed due to STL, meaning normal approximations are generally accurate. [Grubb's Test](https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers) finds outliers in a sample set that is known to be normally distributed. Grubb's test uses the chi-squared distribution to take into account the sample size by means of degrees of freedom to determine outliers to a specific p value. Thermometr's implementation defaults the p value to .025 but it can be overriden in the function call. 

#### ARIMA
An attempt was made to extend the methodology behind S-H-ESD using the ARIMA model which we have used in the past for predicting Mobile DAUs. The ARIMA version is only used when the strict flag is set to true. With the strict flag true, anomalies are only counted if they pass both the ARIMA and STL based tests. The addition of the ARIMA model is an attempt to give the user the ability to minimize false positives.

This algorithm works similarly to the previous, in that we are generating a curve fit and using the errors from the curve to find outliers with Grubb's test. A technique called differencing is used to convert the timeseries into a stationary timeseries (minimal growth trend) in order to better fit the ARIMA model. You can read more about the rationale [here](http://datascienceplus.com/time-series-analysis-building-a-model-on-non-stationary-time-series/).

  Unlike STL, ARIMA does not guarantee errors to be symmetrically or normally distributed. However, ```"Mathematics tell us that linear predictor can be optimal only when the process is Gaussian. When the process is non-Gaussian, a better predictor may be given by a non-linear dynamic model" (Masani and Wiener,  1959)```. In essence, if the errors on the curve aren't normal, than the curve fitting didn't work anyway and curve probably can't be represented with a linear growth time series model such as STL or ARIMA anyway. This also points out an important limitation in this whole process, non linear growth data is going to be significantly less accurate. 

### Notes
* This process doesn't work well for non linear growth since neither ARIMA nor STL are good model fits
* Strict mode is by definition strict, in general this does not need to be used
* Points can become anomalies after the fact, so detect_latest() may not catch an anomaly that when more data is present is flagged, and vice versa. 



### TODO

* Increase support for non daily frequencies
* Add more documentation 


### References:

* Ozaki, T. & Iino, M. An innovation approach to non-Gaussian time series analysis Journal of Applied Probability, 2001, 38, 78-92
* [Twitter S-H-ESD Algorithm](https://blog.twitter.com/2015/introducing-practical-and-robust-anomaly-detection-in-a-time-series)
* [Grubb's Test](https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers) 
