import statsmodels.api as sm
import math
import random
import pandas as pd
from scipy.stats import t
import scipy.stats as st
from numpy import diff
import numpy as np
import collections

from numbers import Number


class Thermometr():
    """ Thermometr objects generate anomoly measurements from input series
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
    def seasonal_esd(inputs, a =.025, frequency = 3):
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
        Returns:
            List of tuple pairs (anomoly,index) indicating the anomolies for input series
        """

        outliers = []
        raw = np.copy(inputs) # copy so that you keep inputs immutable        
        data = sm.tsa.seasonal_decompose(raw, freq=frequency)  # STL decomposition algorithm from stats_model
        vals = data.resid # distance from STL curve for each point in series
        mean = np.nanmean(vals) # mean of the residuals

        # need to impute NULL residuals to mean
        for i in range(len(vals)):
            vals[i]  = mean if np.isnan(vals[i]) else vals[i]
        # run grubbs test on all the items in time order
        for i in range(len(raw)):
            g = 0
            val = 0
            n = len(vals)
            index = 0

            u = np.nanmean(vals)
            s = np.nanstd(vals)
            # find residual with largest z value, or distance from mean
            for j in range(n): 
                val = vals[j] if abs( (u - vals[j]) /s)> g else val
                index = j if abs( (u - vals[j]) /s)> g else index
                g = abs( (u - vals[j]) /s) if abs( (u - vals[j]) /s)> g else g

            # generate critical value for grubb's test
            critical = ( (n -1) /math.sqrt(n))*math.sqrt(math.pow(t.ppf(a/(2*n), n -2),2)/ (n -2 + math.pow(t.ppf(a/(2*n),n-2),2)  )   )
            if g > critical:
                outliers.append((inputs[index],index))
            else:
                check = False
            # remove value if it passes for future test
            temp = np.copy(vals)
            np.delete(temp, index)
            vals[index] = np.nanmean(temp)
        return outliers

    @staticmethod
    def arima_clean(values, known):
        for n in known:
            if n > 13:
                model = sm.tsa.AR(values[:n -1]).fit()
                est = model.predict(n,n+20)
                values[n] = est[0]
            else:
                values[n] = np.nanmean(values[:n-1])
            return values

    @staticmethod
    def arima_anomoly(values, anomoly_index= None, clean= None, margin=1, strict =True ):
        n = len(values) if anomoly_index == None else anomoly_index
        if n < 13:
            return not strict
        model = sm.tsa.AR(values[:n -1]).fit() if not clean is not None else sm.tsa.AR(clean[:n -1]).fit()
        base = 12
        fits = model.predict(base, n)
        errs = []
        for i in range(1, n-base -1):
            errs.append( abs (fits[i] - values[i+base])/values[i+base] )

        errs = np.array(errs)
        u = np.nanmean(errs)
        diff = abs((fits[len(fits)-1] - values[n])/fits[len(fits)-1])
        print u, diff, values[n], fits[len(fits)-1]
        return True if diff >= u and u  else False

    @staticmethod
    def derivative(y):
        dx = 0.001
        dy = diff(y)/dx
        return np.multiply(dy,.001)

    @staticmethod
    def seasonality(y):
        frequency = 2
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
        return int(np.array(counts[1:]).mean()  ) 


    @staticmethod
    def read(series):
        vals = np.copy(series)
        potential = Thermometr.seasonal_esd(vals, frequency= Thermometr.seasonality(vals))
        potential = sorted(potential, key=lambda x: x[1])
        potential_indexes = [x[1] for x in potential]
        clean = Thermometr.arima_clean(vals, potential_indexes)
        confirmed = []
        for x in potential:
            if Thermometr.arima_anomoly(vals, x[1],clean, strict=False):
                confirmed.append(x)
                vals[x[1]] = vals[x[1] -1]
        return confirmed

    def detect(self):
        results = []
        for sub_series in self.series:
            results.append(Thermometr.read(sub_series))
        if self.dates is not None:
            temp = []
            for sub_series in results:
                temp.append ([ (x[0], self.dates[x[1]])  for x in sub_series ])
            results = temp
        if len(results) ==1:
            results = results[0]
        return results
