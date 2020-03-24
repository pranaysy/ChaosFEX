# -*- coding: utf-8 -*-
"""
Spyder Editor

This module contains functions to compute forward trajectory and backward trajectory along the skew-tent map..
"""
import numpy as np
#from numba import vectorize, float64, njit
#@vectorize([float64(float64, float64, nopython=True )])
def skew_tent(VALUE, THRESHOLD, DIRECTION_ITER):
    """
Computes a single step of iteration through the skew-tent map given an
input (previous) value and a threshold. Returns the next value as output.
Based on DIRECTION_ITER the function either computes a single step of forward
trajectory or a single step of backward trajectory.

    Parameters
    ----------
    VALUE : scalar, float64
        Input value to the skew-tent map.
    THRESHOLD : scalar, float64
        Threshold value for the skew-tent map
    DIRECTION_ITER : string
        DIRECTION_ITER == forward computes forward iteration,
        DIRECTION_ITER == backward computes backward iteration

    Returns
    -------
    for  DIRECTION_ITER == forward -> a scalar is returned
    for DIRECTION_ITER == backward -> a vector is returned

    """
    if DIRECTION_ITER == "forward":
        if VALUE < THRESHOLD:
            return VALUE/THRESHOLD

        return (VALUE-1)/(THRESHOLD-1)


    elif DIRECTION_ITER == "backward":
        out = np.zeros((1,2), dtype = np.float64)
        out[0,0] = THRESHOLD * VALUE
        out[0,1]  = (THRESHOLD - 1) * VALUE + 1
        return out
    
#@njit    
def iterations(VALUE, THRESHOLD, LENGTH, DIRECTION_ITER, RANDOM_SEQUENCE):
  """
    
    Parameters
    ----------
    VALUE : scalar, float64
        Initial value of the map
    THRESHOLD : scalar, float64
        Threshold value of the skew-tent map
    LENGTH : scalar, integer
        Length of the chaotic trajectory.
    DIRECTION_ITER : string
        DIRECTION_ITER == forward computes forward iteration,
        DIRECTION_ITER == backward computes backward iteration
    RANDOM_SEQUENCE : array, integer 
        A randomly generated array of ones and zeros

    Returns
    -------
    trajectory : numpy.array, float64
         for  DIRECTION_ITER == forward -> a numpy.array representing the forward trajectory of a chaotic map is returned.
         for DIRECTION_ITER == backward -> a numpy.array representing the backward trajectory of a chaotic map is returned.

    """
  if DIRECTION_ITER == "backward":
    trajectory = np.zeros((LENGTH,1))
    trajectory[0,0] = VALUE
    for i in range(1, LENGTH):
        trajectory[i,0] = skew_tent((trajectory[i-1,0]), THRESHOLD, DIRECTION_ITER)[0, RANDOM_SEQUENCE[i]]
    return trajectory

  elif DIRECTION_ITER == "forward":
    trajectory = np.zeros((LENGTH,1))
    trajectory[0,0] = VALUE
    for i in range(1, LENGTH):
        trajectory[i,0] = skew_tent((trajectory[i-1,0]), THRESHOLD, DIRECTION_ITER)
    return trajectory
