import numpy as np
import numpy as np
from typing import Callable

class VisContSub():
    """
    a class for performing continuum subtraction on visibility data
    """
    def __init__(self, fit_func: Callable):
        """ Fit contiuum basline to visibility data

        Args:
            fit_func (Callable): Function to model contiuum baseline
        """

        self.fit_func = fit_func

    def vis_cont_sub(self, visdata: np.ndarray,
                    flags: np.ndarray, weights: np.ndarray):
        """
        Fits the visibility data with the desired function and returns the continuum subtracted data
        """
        ntime, nbaseline, _, ncorr = visdata.shape
        
        for tme in range(ntime):
            for bl in range(nbaseline):
                for corr in range(ncorr):
                    slc = tme, bl, slice(None), corr
                    vis_slice = visdata[slc]
                    flag_slice = flags[slc]
                    weights_slice = weights[slc]
                
                    real_model = self.fit_func.fit(vis_slice.real, flag_slice, weights_slice)
                    imag_model = self.fit_func.fit(vis_slice.imag, flag_slice, weights_slice)

                    visdata[slc] = real_model + 1j * imag_model

        return visdata
