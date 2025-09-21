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
    def __init__(self, order, velwidth, cont_tol):
        """

        Args:
            order (_type_): _description_
            velwidth (_type_): _description_
        """
        self.order = order
        self.velwidth = velwidth
        self.preped = False
        self.cont_tol = cont_tol
    
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

        if not self.preped:
            self.prepare(x)
            
        # Mask invalid or zero-weight points
        mask = (weights > 0) & np.isfinite(x) & np.isfinite(data) & np.isfinite(weights)
        x_masked = x[mask]
        data_masked = data[mask]
        weights_masked = weights[mask]
    
    
        if len(x_masked) < (self.order + 1):
            #TODO(Sphe) maybe return raise an exception, and let the caller of this function decide what to do.
            print("Not enough valid points for spline fit, returning original data.")
            return np.copy(data)  # fallback: just return input
    
        nchan = len(x_masked)
        knotind = np.linspace(0, nchan, self.imax, dtype=int)[1:-1]
        chwid = max(1, (nchan // self.imax) // 8)
        knots_idx = self.rng.integers(-chwid, chwid, size=knotind.shape) + knotind
        knots_idx = np.clip(knots_idx, 1, nchan - 20)  # avoid edges
        knot_positions = np.unique(x_masked[knots_idx])
    
        splCfs = splrep(x_masked, data_masked, task=-1,
                        w=weights_masked, t = knot_positions, k=self.order)
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
        
    def prepare(self, x):
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
        if not (mask is None):
            data[np.logical_not(mask)] = np.nan
        
        if self._imax%2 ==0:
            window = self._imax + 1
        else:
            window = self._imax
        
        padded_data = np.pad(data, window//2, mode="linear_ramp")
        filtered = np.nanmedian(np.lib.stride_tricks.sliding_window_view(padded_data, window), axis = 1)
        
        return filtered

class FitPolynomial(FitFunc):
    """
    Polynomial fitting function using numpy.polyfit
    """
    def __init__(self, order, cont_tol = 0):
        """
        order (int): Order/degree of the polynomial
        """
        self.order = order
        self.preped = False
        self.cont_tol = cont_tol
        
    def prepare(self, x):
        """
        Prepare for polynomial fitting 
        """
        nchan = len(x)
        log.info(f"Polynomial fitting: nchan = {nchan}, order = {self.order}")
        self.preped = True
    
    def fit(self, x, data, weights=None):
        if not self.preped:
            self.prepare(x)

        if weights is not None:
            mask = (weights > 0) & np.isfinite(x) & np.isfinite(data) & np.isfinite(weights)
            x_masked = x[mask]
            data_masked = data[mask]
            weights_masked = weights[mask]
            
        
        else:
            mask = np.isfinite(x) & np.isfinite(data)
            x_masked = x[mask]
            data_masked = data[mask]
            weights_masked = None
            
        if (len(x_masked)/len(x))*100 <= self.cont_tol:
            return np.zeros_like(data)

        try:
            if weights_masked is not None:
                coeffs = np.polyfit(x_masked, data_masked, self.order, w=weights_masked)
            else:
                coeffs = np.polyfit(x_masked, data_masked, self.order)
            return np.polyval(coeffs, x)
        except Exception as e:
            print(f"Polynomial fitting failed: {e}")
        return np.copy(data)