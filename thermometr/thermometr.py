import statsmodels.api as sm
import math
import random
import pandas as pd
from scipy.stats import t
import scipy.stats as st
import numpy as np
import collections

from numbers import Number


class Thermometr():
    """ 
    Thermometr objects generate anomoly measurements from input series
    """


    def __init__(self, series, dates =None):
        """
        Note:
            list comprehension standardizes series to np.arrays

        Args:
            series (Optional[int] or DataFrame or Dict): 1D or 2D group  of numerical values or Dataframe 
            dates [str]: List of dates corresponding to series. Multiple formats are supported
        """
        self.series = [np.array(x) for x in Thermometr.standardize_data(series)]
        self.dates = dates

    @staticmethod
    def standardize_data(series):
        """
        Static method that regularlizes series to 2D list of numpy arrays
        Note:
            Type errors that aren't in the first position of the iterable won't be caught
        Args:
            series (Optional[int] or DataFrame or Dict): 1D or 2D group  of numerical values or Dataframe
        Returns:
            List of numpy arrays containing series, raises ValueError if problem
        """

        lists = []

        if len(series) == 0: 
            raise ValueError('Empty series not allowed')
        # python strings will pass iterable so need to explicitely check
        if isinstance(series,str):
            raise ValueError('Series could not be parsed, must be iterable')
        # DataFrame isn't a standard iterable, .values returns np.array()
        if isinstance(series, pd.DataFrame):
            for i in df.columns:
                lists.append(df[i].values)
            return lists
        # can't select from dict with [0] index so need seperate case
        elif isinstance(series, dict):
            for sub in series.values():
                lists.extend(Thermometr.standardize_data(sub))
            return lists
        # standard use case
        elif isinstance(series, collections.Iterable):
            if isinstance(series[0], Number):
                lists.append(series)
                return lists
            # if it is a list of lists, then it recurses into this function again
            elif isinstance(series[0], collections.Iterable):
                for sub in series:
                    lists.extend(Thermometr.standardize_data(sub))
                return lists

        raise ValueError('Series could not be parsed, must be iterable')        

    @staticmethod
    def seasonal_esd(inputs, a =.025, frequency = 3, start= None, end =None):
        """
        Static method that generates anomolies from the given series
            using S-H-ESD algorithm developed by twitter
        Note:
            Based on frequency, the decompisition trend values can become NULL near the bounds of the series
                this is a stats_model limitation and has been dealt with using an imputation
        Args:
            inputs (np.array[int]): time series of numerical values
            a (float): a confidence level(alpha) for the algorithm to use to determine outliers
            frequency (int): the frequency of season of the data (instances to complete cycle) 
            start (int): the first index in the series to check for anomalies
            end (int): the last index in the series to check for anomalies
        Returns:
            List of tuple pairs (anomoly,index) indicating the anomolies for input series
        """

        outliers = []
        raw = np.copy(inputs) # copy so that you keep inputs immutable        
        data = sm.tsa.seasonal_decompose(raw, freq=frequency)  # STL decomposition algorithm from stats_model
        
        trend = data.trend
        for i in range(len(trend)):
            substitute = trend[i-frequency] if i > frequency else trend[i + frequency] 
            trend[i]  = substitute if np.isnan(trend[i]) else trend[i]
        vals = data.observed - data.trend - data.seasonal  # distance from STL curve for each point in series
        mean = np.nanmean(vals) # mean of the residuals 

        # need to impute NULL residuals to mean
        pairs = []
        for i in range(len(vals)):
            v  = mean if np.isnan(vals[i]) else vals[i]
            pairs.append((v,i))
        return Thermometr.grubbs_test(inputs, pairs, start,end,a)

    @staticmethod 
    def grubbs_test( inputs,errs, start=None,end=None, a = .025):
        """
        Static method that finds anomalies in a sample of normally distributed random variables

        Note:
          An assumption is made that the input data is normal. Grubb's test uses the chi squared 
          distribution and n degrees of freedom to account for small sample sizes

        Args:
            inputs (np.array[int]): time series of numerical values
            start (int): the first index in the series to check for anomalies
            end (int): the last index in the series to check for anomalies
            a (float): a confidence level(alpha) for the algorithm to use to determine outliers
        """
        outliers = []
        vals = errs
        start = 0 if start is None else start
        end = len(vals) -1  if end is None else end
        check = True

        # run grubbs test until the furthest remaining point fails
        while check == True:
            g = 0
            val = 0
            n = len(vals)
            index = 0
            ind = 0
            series  = [ x[0] for x in vals]
            series = np.array(series)
            u = np.nanmean(series)
            s = np.nanstd(series)
            # find residual with largest z value, or distance from mean
            for j in range(len(vals)):
                if j >= start and j <= end:
                    v = vals[j][0]
                    k = vals[j][1]
                    val = v if abs( (u - v )/s)> g else val
                    index = int(j) if abs( (u - v) /s)> g else index
                    ind = k if abs( (u - v) /s)> g else ind
                    g = abs( (u - v) /s) if abs( (u - v) /s)> g else g
    
            # generate critical value for grubb's test
            critical = ( (n -1) /math.sqrt(n))*math.sqrt(math.pow(t.ppf(a/(2*n), n -2),2)/ (n -2 + math.pow(t.ppf(a/(2*n),n-2),2)  )   )
		
            if g > critical:
                outliers.append((inputs[ind],ind, 1 - critical/g))
            else:
                check = False
            # remove value for next iteration of the test by imputing to new mean
            vals[index] = (u*n - vals[index])/ (n-1)
        return outliers

    @staticmethod
    def arima_test(values, clean,start= None, end=1, strict =True ):
        """
        Static method that is used for finding anomalies with ARIMA and not STL
        Note:
             ARIMA doesn't work on small series so in strict mode validation defaults to false
             and in non strict defaults to true
        Args:
            values (np.array[int]): time series values
            start (int): the first index in the series to check for anomalies
            end (int): the last index in the series to check for anomalies
            strict (binary): determines the default return value for when a comparison cannot be completed 
        Returns:
            List of tuple pairs (anomoly,index) indicating the anomolies for input series
        """
        try:
            n = len(values)
            # ARIMA doesn't work on series less than 13
            if n < 13:
                return not strict
            # Build the ARIMA model and generate projections
            model = sm.tsa.AR(clean).fit()
            base = 12 # ARIMA requires starting the projections at an index less than the maximum
            fits = model.predict(base, n)
            errs = []
            # generates all the differences between arima projections and observed values
            for i in range(base,n):
                errs.append((fits[i-base] - values[i],i))

            return Thermometr.grubbs_test(values, errs)
        except Exception as e:
            return []


    @staticmethod
    def derivative(y):
        """
        Returns deriviative of series y
        
        Args:
            y (np.array[float]): series of values
        Returns
            np.array[float] of changes between values 
        """
        dx = 0.001
        dy = np.diff(y)/dx
        return np.multiply(dy,.001)

    @staticmethod
    def seasonality(y):
        """
        Generates the frequency of a timeseries
        Notes:
           computes frequency by checking average number of times
           for the derivative to change sign 3 times
        Args:
            y (np.array[float]): series of values
        Returns
            int representing frequency 
        """
        vals = Thermometr.derivative(y)
        sign = 1 if vals[0] >=0 else -1
        changes = 0
        count = 0
        counts = [] 
        for val in vals:
            count+=1
            if val * sign < 0:
                sign = -1 if sign >=0 else 1
                changes +=1
                if changes ==3:
                    changes =0
                    counts.append(count)
                    count = 0
        try:
            return int(np.nanmean(np.array(counts[1:]) ) ) 
        except Exception as e:
            print e
            return 3

    @staticmethod
    def detect_series(series,start,end, strict = False , a=0.025):
        """
        Finds anomalies in a series using ESD and ARIMA validation
        Args:
            series (np.array[float]): the time series values
            start (int): the first index in the series to check for anomalies
            end (int): the last index in the series to check for anomalies
        Returns [tuples] containing anomalies, their index, their value, and their ESD score
        """
        vals = np.copy(series)
        potential = Thermometr.seasonal_esd(vals,a=a,frequency= Thermometr.seasonality(vals), start=start, end=end)
        potential = sorted(potential, key=lambda x: x[1])
        indices = [x[1] for x in potential]
 
        clean = np.copy(vals)
        for i in range(len(vals)):
            if i in indices and i >0:
                clean[i] = clean[i-1]

        # check against ARIMA results, if strict mode only take intersection of anomaly sets
        others = Thermometr.arima_test(vals,clean)
        if strict:
            confirmed = []
            indices = [x[1] for x in others]
            for x in potential:
                if x[1] < 12 or x[1] in indices:
                    confirmed.append(x)
            potential = confirmed
        else:
            potential.extend(others)
        potential  = {v[1]:v for v in potential}.values()
        anomalies = sorted(potential, key=lambda x: x[1])
        return anomalies

    def detect(self, start = None, end = None, strict = False, a =0.025):
        """
        Finds anomalies in each series of Thermometr
        Args:
            start (int): the first index in the series to check for anomalies
            end (int): the last index in the series to check for anomalies
        Returns:
             list or list of lists contianing anomalies for Thermometr
        """
        results = []

        for sub_series in self.series:
            n = len(sub_series)
            s= eval(start) if type(start) == str else start
            e = eval(end) if type(end) == str else end
            results.append(Thermometr.detect_series(sub_series,s,e,strict,a))
        if self.dates is not None:
            temp = []
            for sub_series in results:
                temp.append ([ {"value":x[0], "index":self.dates[x[1]], "ESD":x[2]}  for x in sub_series ])
            results = temp
        else:
            temp = []
            for sub_series in results:
                temp.append ([ {"value":x[0], "index":x[1], "ESD":x[2]}  for x in sub_series ])
            results = temp
        if len(results) ==1:
            results = results[0]
        return results

    def detect_latest(self,strict = False, a=0.025):
        """
        Finds last index anomalies in each series of Thermometr
        Notes:
            same as detect() except sets start = n-1 and end = n given n = len(series)
        Returns:
             list or list of lists contianing anomalies for Thermometr
        """
        return self.detect("n-1","n", strict,a)
