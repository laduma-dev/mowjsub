from scipy.interpolate import splev, splrep
import sys
from scabha import init_logger
from abc import ABC, abstractmethod
from . import BIN
import numpy as np

log = init_logger(BIN.im_plane)


class FitFunc:
    """
    abstract class for writing fitting functions
    """
    def __init__(self, order, velwidth):
        """

        Args:
            order (_type_): _description_
            velwidth (_type_): _description_
        """
        self.order = order
        self.velwidth = velwidth
        self.preped = False
    
    def prepare(self, x):
        nchan = len(x)
        msort = np.argpartition(x, -2)
        m1l, m2l = msort[-2:]
        m1h, m2h = msort[:2]
        if np.abs(m1l - m2l) == 1 and np.abs(m1h - m2h) == 1:
            dvl = np.abs(x[m1l]-x[m2l])/np.mean([x[m1l],x[m2l]])*3e5
            dvh = np.abs(x[m1h]-x[m2h])/np.mean([x[m1h],x[m2h]])*3e5
            self.dv = (dvl+dvh)/2
            self.imax = int(nchan / (self.velwidth//self.dv))+1
        else:
            log.error('The frequency values are not changing monotonically, aborting')
            sys.exit(1)
            
        
        log.info(f"nchan = {nchan}, dv = {self.dv}, {self.velwidth}km/s in chans:"
                f" {self.velwidth//self.dv}, max order spline = {self.imax}")
        self.preped = True

    
    @abstractmethod
    def fit(self, x, data, mask, weight):
        pass
    
class FitBSpline(FitFunc):
    """
    BSpline fitting function based on `splev`, `splrep` in `scipy.interpolate` 
    """
    def __init__(self, order, velWidth, randomState=None, seq=None):
        """
        needs to know the order of the spline and the number of knots
        """
        self.order = order
        self.velwidth = velWidth
        self.preped = False
        if randomState and seq:
            rs = np.random.SeedSequence(entropy = randomState, spawn_key = (seq,))
        else:
            rs = np.random.SeedSequence()
        self.rng = np.random.default_rng(rs)
        
        
    def fit(self, x, data, weights):
        """
        returns the spline fit and the residuals from the fit
        
        x : x values for the fit
        data : values to be fit by spline
        weight : weights for fitting the Spline. 
            To mask values, set the corresponding weight to zero.
        """
        nchan = len(x)
        knotind = np.linspace(0, nchan, self.imax, dtype = int)[1:-1]
        chwid = (nchan // self.imax) // 8
        knots = lambda: self.rng.integers(-chwid, chwid, size = knotind.shape)+knotind
        
        
        splCfs = splrep(x, data, task = -1, w = weights, t = x[knots()], k = self.order)
        spl = splev(x, splCfs)
        return spl

class FitMedFilter(FitFunc):
    """
    Median filtering class for continuum subtraction 
    """
    def __init__(self, velWidth):
        """
        needs to know the order of the spline and the number of knots
        """
        self._velwid = velWidth
        
    def prepare(self, x, data = None, mask = None, weight = None):
        msort = np.argpartition(x, -2)
        m1l, m2l = msort[-2:]
        m1h, m2h = msort[:2]
        if np.abs(m1l - m2l) == 1 and np.abs(m1h - m2h) == 1:
            dvl = np.abs(x[m1l]-x[m2l])/np.mean([x[m1l],x[m2l]])*3e5
            dvh = np.abs(x[m1h]-x[m2h])/np.mean([x[m1h],x[m2h]])*3e5
            dv = (dvl+dvh)/2
            self._imax = int(self._velwid//dv)
            if self._imax %2 == 0:
                self._imax += 1
            log.info('len(x) = {}, dv = {}, {}km/s in chans: {}'.format(len(x), dv, self._velwid, self._velwid//dv))
        else:
            log.debug('probably x values are not changing monotonically, aborting')
            sys.exit(1)
            
    
    def fit(self, x, data, mask, weight):
        """
        returns the median filtered data as line emission
        
        x : x values for the fit
        y : values to be fit
        mask : a mask (not implemented really)
        weight : weights
        """
        cp_data = np.copy(data)
        if not (mask is None):
            data[np.logical_not(mask)] = np.nan
        nandata = np.hstack((np.full(self._imax//2, np.nan), data, np.full(self._imax//2, np.nan)))
        nanMed = np.nanmedian(np.lib.stride_tricks.sliding_window_view(nandata,self._imax), axis = 1)
        # resMed = nanMed[~np.isnan(nanMed)]
        resMed = nanMed
        return resMed, cp_data-resMed
