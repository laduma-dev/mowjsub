import numpy as np
from scabha import init_logger
from . import BIN
from .exceptions import BadFitError


log = init_logger(BIN.im_plane)

class ContSub():
    """
    a class for performing continuum subtraction on data
    """
    def __init__(self, fit_func, nomask=False):
        """
        each object can be initiliazed by passing a data cube, a fitting function, and a mask
        Args:
            fit_func (Callable) : a fitting function should be built on FitFunc class
        """
        self.fit_func = fit_func
        self.nomask = nomask 
        
        
    def fitContinuum(self, cube, mask):
        """
        fits the data with the desired function and returns the continuum and the line
        
        Args:
            cube (Array): Data cube to subtract continuum from
            mask (Array): Binary data weights. True -> will be used in fit, False will not be used in fit.

        Returns:
            Array: Continuum fit
        """
        
        dimx, dimy, _ = cube.shape
            
        cont_model = np.zeros_like(cube)

        if isinstance(mask, np.ndarray):
            nomask = False
            mask = np.asarray(mask, dtype=bool)
        else:
            nomask = True
            
        fitfunc = self.fit_func

        skipped_lines = 0 
        for ra in range(dimx):
            for dec in range(dimy):
                
                # slice the data cube for the current pixel
                slc = ra,dec,slice(None)
                cube_ij = cube[slc]
                if nomask:
                    mask_ij = np.full_like(cube_ij, False, dtype=bool)
                else:
                    mask_ij = mask[slc]
                
                try:
                    cont_model[slc] = fitfunc.fit(cube_ij, mask_ij, weights=None)
                except BadFitError:
                    # Flag LOS and continue if too many pixels are flagged
                    cont_model[slc] = np.full_like(cube_ij, np.nan)
                    skipped_lines += 1
        
        if skipped_lines > 0:
            log.info(f"This worker set {skipped_lines} spectra to NaN because of --cont-fit-tol.")
        
        return cont_model
    