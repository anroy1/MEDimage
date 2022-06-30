#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import iqr, kurtosis, skew, scoreatpercentile, variation

def mean(vol: np.ndarray) -> float:
    """Compute statistical mean feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: Statistical mean feature
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(x)  # Mean

def var(vol: np.ndarray) -> float:
    """Compute statistical variance feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: Statistical variance feature
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.var(x)  # Variance

def skewness(vol: np.ndarray) -> float:
    """Compute the sample skewness feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: The skewness feature of values along an axis. Returning 0 where all values are
        equal.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return skew(x)  # Skewness

def kurt(vol: np.ndarray) -> float:
    """Compute the kurtosis (Fisher or Pearson) feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: The kurtosis feature of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return kurtosis(x)  # Kurtosis

def median(vol: np.ndarray) -> float:
    """Compute the median feature along the specified axis of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: The median feature of the array elements.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.median(x)  # Median

def min(vol: np.ndarray) -> float:
    """Compute the minimum grey level feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: The minimum grey level feature of the array elements.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.min(x)  # Minimum grey level

def max(vol: np.ndarray) -> float:
    """Compute the maximum grey level feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: The maximum grey level feature of the array elements.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.max(x)  # Maximum grey level

def P10(vol: np.ndarray) -> float:
    """Calculate the score at the 10th percentile feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: Score at 10th percentil.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 10)  # 10th percentile

def P90(vol: np.ndarray) -> float:
    """Calculate the score at the 90th percentile feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: Score at 90th percentil.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 90)  # 90th percentile

def iqrange(vol: np.ndarray) -> float:
    """Compute the interquartile range feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: Interquartile range. If axis != None, the output data-type is the same as that of the input.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return iqr(x)  # Interquartile range

def range(vol: np.ndarray) -> float:
    """Range of values (maximum - minimum) feature along an axis of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: A new array holding the range of values, unless out was specified, in which case a reference to out is returned.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.ptp(x)  # Range max(x) - min(x) 

def mad(vol: np.ndarray) -> float:
    """Mean absolute deviation feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float : A new array holding mean absolute deviation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(np.absolute(x - np.mean(x)))  # Mean absolute deviation

def rmad(vol: np.ndarray) -> float:
    """Robust mean absolute deviation feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
        P10(ndarray): Score at 10th percentil.
        P90(ndarray): Score at 90th percentil.
    
    Returns:
        float: A new array holding the robust mean absolute deviation.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization
    P10 = scoreatpercentile(x, 10)  # 10th percentile
    P90 = scoreatpercentile(x, 90)  # 90th percentile
    x_10_90 = x[np.where((x >= P10) &
                         (x <= P90), True, False)]  # Holding x for (x >= P10) and (x<= P90)
                         
    return np.mean(np.abs(x_10_90 - np.mean(x_10_90)))  # Robust mean absolute deviation

def medad(vol: np.ndarray) -> float:
    """Median absolute deviation feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
    
    Returns:
        float: A new array holding the median absolute deviation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(np.absolute(x - np.median(x)))  # Median absolute deviation

def cov(vol: np.ndarray) -> float:
    """Compute the coefficient of variation feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
   
    Returns:
        float: A new array holding the coefficient of variation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return variation(x)  # Coefficient of variation

def qcod(vol: np.ndarray) -> float:
    """Compute the quartile coefficient of dispersion feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
  
    Returns:
        float: A new array holding the quartile coefficient of dispersion feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization
    x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)  

    return iqr(x) / x_75_25  # Quartile coefficient of dispersion

def energy(vol: np.ndarray) -> float:
    """Compute the energy feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
  
    Returns:
        float: A new array holding the energy feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.sum(np.power(x, 2))  # Energy

def rms(vol: np.ndarray) -> float:
    """Compute the root mean square feature of the input dataset (3D Array).

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
             --> vol: continuous imaging intensity distribution
  
    Returns:
        float: A new array holding the root mean square feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.sqrt(np.mean(np.power(x, 2)))  # Root mean square

def extract_all(vol: np.ndarray, intensity: str = None) -> dict:
    """Compute stats.

     Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            --> vol: continuos imaging intensity distribution
        intensity(optional str): If 'arbitrary', some feature will not be computed.
            If 'definite', all feature will be computed. If not present as an argument,
            all features will be computed. Here, 'filter' is the same as
            'arbitrary'.
 
    Return:
        dict: Dictionnary containing all stats features.
    """
    # PRELIMINARY
    if intensity is None or intensity == 'definite':
        definite = True
    elif intensity == 'arbitrary' or intensity == 'filter':
        definite = False
    else:
        raise ValueError('Second argument must either be "arbitrary" or \
                         "definite" or "filter"')

    x = vol[~np.isnan(vol[:])]  # Initialization

    # Initialization of final structure (Dictionary) containing all features.
    stats = {'Fstat_mean': [],
             'Fstat_var': [],
             'Fstat_skew': [],
             'Fstat_kurt': [],
             'Fstat_median': [],
             'Fstat_min': [],
             'Fstat_P10': [],
             'Fstat_P90': [],
             'Fstat_max': [],
             'Fstat_iqr': [],
             'Fstat_range': [],
             'Fstat_mad': [],
             'Fstat_rmad': [],
             'Fstat_medad': [],
             'Fstat_cov': [],
             'Fstat_qcod': [],
             'Fstat_energy': [],
             'Fstat_rms': []
             }

    # STARTING COMPUTATION
    if definite:
        stats['Fstat_mean'] = np.mean(x)  # Mean
        stats['Fstat_var'] = np.var(x)  # Variance
        stats['Fstat_skew'] = skew(x)  # Skewness
        stats['Fstat_kurt'] = kurtosis(x)  # Kurtosis
        stats['Fstat_median'] = np.median(x)  # Median
        stats['Fstat_min'] = np.min(x)  # Minimum grey level
        stats['Fstat_P10'] = scoreatpercentile(x, 10)  # 10th percentile
        stats['Fstat_P90'] = scoreatpercentile(x, 90)  # 90th percentile
        stats['Fstat_max'] = np.max(x)  # Maximum grey level
        stats['Fstat_iqr'] = iqr(x)  # Interquantile range
        stats['Fstat_range'] = np.ptp(x)  # Range max(x) - min(x)
        stats['Fstat_mad'] = np.mean(np.absolute(x - np.mean(x)))  # Mean absolute deviation
        x_10_90 = x[np.where((x >= stats['Fstat_P10']) &
                             (x <= stats['Fstat_P90']), True, False)]
        stats['Fstat_rmad'] = np.mean(np.abs(x_10_90 - np.mean(x_10_90)))  # Robust mean absolute deviation
        stats['Fstat_medad'] = np.mean(np.absolute(x - np.median(x)))  # Median absolute deviation
        stats['Fstat_cov'] = variation(x)  # Coefficient of variation
        x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)
        stats['Fstat_qcod'] = iqr(x)/x_75_25  # Quartile coefficient of dispersion
        stats['Fstat_energy'] = np.sum(np.power(x, 2))  # Energy
        stats['Fstat_rms'] = np.sqrt(np.mean(np.power(x, 2)))  # Root mean square

    return stats
