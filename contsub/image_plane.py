import numpy as np
from scabha import init_logger
from . import BIN


log = init_logger(BIN.im_plane)

class ContSub():
    """
    a class for performing continuum subtraction on data
    """
    def __init__(self, fit_func, nomask, fit_tol=0):
        """
        each object can be initiliazed by passing a data cube, a fitting function, and a mask
        Args:
            fit_func (Callable) : a fitting function should be built on FitFunc class
            nomask (bool): Ignore mask if set
            fit_tol (float): If set, will skip lines with more than this percentage of masked pixels.
                             Default is 0, which means no skipping.
                             If set to 100, will skip all lines with any masked pixels.
                             If set to 0, will not skip any lines.
        Returns:
            None
        """
        self.nomask = nomask
        self.fit_func = fit_func
        self.fit_tol = fit_tol
        
        
    def fitContinuum(self, xspec, cube, mask):
        """
        fits the data with the desired function and returns the continuum and the line
        
        Args:
            xspec (Array): Spectrum coordinates
            cube (Array): Data cube to subtract continuum from
            mask (Array): Binary data weights. True -> will be used in fir, False will not be used in fit.

        Returns:
            Array: Continuum fit
        """
        
        dimx, dimy, nchan = cube.shape
            
        cont_model = np.zeros_like(cube)
        nomask = self.nomask
        if nomask:
            mask = None
            
        fitfunc = self.fit_func
        if not fitfunc.preped:
            fitfunc.prepare(xspec)

        skipped_lines = 0 
        for ra in range(dimx):
            for dec in range(dimy):
                # slice the data cube for the current pixel
                slc = ra,dec,slice(None)
                mask_ij = mask[slc] if nomask == False else None
                cube_ij = cube[slc]
                
                # Find and mask NaNs in the data
                nanvals_idx = np.where(np.isnan(cube_ij))
                nansize = len(nanvals_idx[0])
                if nansize == nchan:
                    cont_model[slc] = np.full_like(cube_ij, np.nan)
                    continue
                elif nansize > 0:
                    if nomask:
                        mask_ij = np.ones_like(cube_ij)
                    mask_ij[nanvals_idx] = 0
                
                # Flag LOS and continue if too many pixels are flagged
                if self.fit_tol > 0:
                    if isinstance(mask_ij, np.ndarray) and \
                            (nchan - mask_ij.sum()) / nchan > self.fit_tol/100:
                        skipped_lines += 1
                        cont_model[slc] = np.full_like(cube_ij, np.nan)
                        continue
                
                cont_model[slc] = fitfunc.fit(xspec, cube_ij, 
                                                weights = mask_ij)
        
        if skipped_lines > 0:
            log.info(f"This worker set {skipped_lines} spectra to NaN because of --cont-fit-tol.")
            
        return cont_model
    