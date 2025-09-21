from contsub.fitfuncs import FitBSpline
import numpy as np
import xarray as xr
import dask.array as da
import numpy as np
from typing import Callable

class VisContSub():
    """
    a class for performing continuum subtraction on visibility data
    """
    def __init__(self, fit_func: Callable, fit_tol: float=0):
        """
        Args:
            method (str): Fitting method to use, e.g., 'spline'
            order (int): Order of the spline fit
            vel_width (float): Velocity width for the fit
        """

        self.fit_func = fit_func
        self.fit_tol = fit_tol
        
        

    def vis_cont_sub(self, xspec: np.ndarray, visdata: np.ndarray,
                     flags: np.ndarray, weights: np.ndarray):
        """
        Fits the visibility data with the desired function and returns the continuum subtracted data
        """
        
        ntime, nbaseline, nchan, ncorr = visdata.shape
        
        for tme in range(ntime):
            for bl in range(nbaseline):
                for corr in range(ncorr):
                    vis_slice = visdata[tme, bl, :, corr]
                    flag_slice = flags[tme, bl, :, corr]
                    weights_slice = weights[tme, bl, :, corr]
                
                    weights_slice[flag_slice] = 0  
                    real_model = self.fit_func.fit(xspec, vis_slice.real, weights_slice)
                    imag_model = self.fit_func.fit(xspec, vis_slice.imag, weights_slice)

                    visdata[tme,bl,:,corr] = real_model + 1j * imag_model

        return visdata


