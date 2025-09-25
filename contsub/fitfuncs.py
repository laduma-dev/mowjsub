from scipy.interpolate import splev, splrep
import sys
from scabha import init_logger
from abc import abstractmethod
from . import BIN
from .exceptions import BadFitError
import numpy as np
from scipy import fftpack

log = init_logger(BIN.im_plane)

class FitFunc:
    """
    abstract class for writing fitting functions
    """
    def __init__(self, freqs, velwidth:float, fit_tol:float=0):
        """

        Args:
            order (_type_): _description_
            velwidth (_type_): _description_
        """
        self.velwidth = velwidth
        self.fit_tol = fit_tol
        self.freqs = freqs
        self.nchan = freqs.size

    def prepare(self):
        freqs = self.freqs
        msort = np.argpartition(freqs, -2)
        m1l, m2l = msort[-2:]
        m1h, m2h = msort[:2]
        if np.abs(m1l - m2l) == 1 and np.abs(m1h - m2h) == 1:
            dvl = np.abs(freqs[m1l]-freqs[m2l])/np.mean([freqs[m1l],freqs[m2l]])*3e5
            dvh = np.abs(freqs[m1h]-freqs[m2h])/np.mean([freqs[m1h],freqs[m2h]])*2.998e5
            dv = (dvl+dvh)/2
            self.imax = int(self.velwidth/dv)
            if self.imax %2 == 0:
                self.imax += 1
            log.info(f"nchan = {self.nchan}, dv = {dv}, {self.velwidth}km/s in chans: {self.imax}")
        else:
            log.error('probably x values are not changing monotonically, aborting')
            sys.exit(1)
        self.preped = True

    def invalid_point_count(self, data:np.ndarray, mask:np.ndarray):
        """ Calculates the number of invalid data points in a spectrum.

        Args:
            data (np.ndarray): 1D spectrum
            mask (np.ndarray): Binary mask
        """
        data_count = np.isnan(data).sum()
        mask_count = mask.sum()
        
        return data_count + mask_count
    
    def is_fit_possible(self, data: np.ndarray, mask: np.ndarray, raise_exception=True):
        """_summary_

        Args:
            data (np.ndarray): _description_
            mask (np.ndarray): _description_
        """
        invalid = self.invalid_point_count(data, mask)
        nchan = data.size
        valid_fraction = (1 - invalid / nchan) * 100
        if valid_fraction < self.fit_tol:
            if raise_exception:
                raise BadFitError(f"Fraction of valid data points {valid_fraction:.2f} is less than the required tolerance (--cont-fit-tol {self.fit_tol})")
            else:
                return False
        else:
            return True
    
    @abstractmethod
    def fit(self, x, data, mask, weight):
        pass
    
class FitBSpline(FitFunc):
    """
    BSpline fitting function based on `splev`, `splrep` in `scipy.interpolate` 
    """
    def __init__(self, freqs, order, velwidth, fit_tol=0, randomState=None, seq=None):
        """
        needs to know the order of the spline and the number of knots
        """
        self.order = order
        self.velwidth = velwidth
        self.fit_tol = fit_tol
        self.freqs = freqs
        self.nchan = freqs.size
        self.prepare() 
        self.max_spline_order = int(self.nchan / self.imax) + 1
        log.info(f"max spline order: {self.max_spline_order}")

        if randomState and seq:
            rs = np.random.SeedSequence(entropy = randomState, spawn_key = (seq,))
        else:
            rs = np.random.SeedSequence()
        self.rng = np.random.default_rng(rs)
        
        
    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the spline fit and the residuals from the fit
        
        x : x values for the fit
        data : values to be fit by spline
        weight : weights for fitting the Spline. 
            To mask values, set the corresponding weight to zero.
        """

        self.is_fit_possible(data, mask, raise_exception=True)
        nvalid = self.nchan - self.invalid_point_count(data, mask)
        
        if nvalid < (self.order + 1):
            raise BadFitError("Not enough valid points for spline fit, returning original data.")
            
        # Mask invalid or zero-weight points
        mask[np.where(np.isnan(data))] = True
        x_masked = self.freqs[~mask] 
        data_masked = data[~mask]
    
        knotind = np.linspace(0, x_masked.size, self.max_spline_order, dtype=int)[1:-1]
        chwid = max(1, (self.nchan // self.max_spline_order) // 8)
        knots_idx = self.rng.integers(-chwid, chwid, size=knotind.shape) + knotind
        knots_idx = np.unique(np.clip(knots_idx, 1, x_masked.size-1)) # avoid edges
        
        knot_positions = x_masked[knots_idx]
        
        if isinstance(weights, np.ndarray):
            splCfs = splrep(x_masked, data_masked, task=-1,
                            w=weights[~mask], t = knot_positions, k=self.order)
        else:
            splCfs = splrep(x_masked, data_masked, task=-1,
                            t=knot_positions, k=self.order)

        return  splev(self.freqs, splCfs)


class FitMedFilter(FitFunc):
    """
    Median filtering class for continuum subtraction 
    """
    def __init__(self, freqs, velwidth, fit_tol=0):
        """
        needs to know the order of the spline and the number of knots
        """
        self.velwidth = velwidth
        self.fit_tol = fit_tol
        self.freqs = freqs
        self.nchan = freqs.size
        self.prepare()
    
            
    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the median filtered data as line emission
        
        data (np.ndarray) : values to be fit
        mask (np.ndarray) : a mask (not implemented really)
        weight (np.ndarray) : weights
        """
        self.is_fit_possible(data, mask, raise_exception=True)
        
        if isinstance(mask, np.ndarray):
            data[mask] = np.nan
        
        padded_data = np.pad(data, self.imax//2, mode="linear_ramp")
        filtered = np.nanmedian(np.lib.stride_tricks.sliding_window_view(padded_data, self.imax), axis = 1)
        
        return filtered

class FitPolynomial(FitFunc):
    """
    Polynomial fitting function using numpy.polyfit
    """
    def __init__(self, freqs, order, fit_tol=0):
        """
        order (int): Order/degree of the polynomial
        """
        self.order = order
        self.fit_tol = fit_tol
        self.freqs = freqs
        self.nchan = freqs.size
        self.prepare()
        
    def prepare(self):
        """
        Prepare for polynomial fitting 
        """
        
        log.info(f"Polynomial fitting: nchan = {self.nchan}, order = {self.order}")
        self.preped = True 
    
    def fit(self, data: np.ndarray, mask:np.ndarray, weights: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): _description_
            data (np.ndarray): _description_
            weights (np.ndarray): _description_
            mask (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        
        self.is_fit_possible(data, mask, raise_exception=True)
        
        
        x_masked = self.freqs[~mask]
        data_masked = data[~mask]
            
        try:
            if isinstance(weights, np.ndarray):
                coeffs = np.polyfit(x_masked, data_masked, self.order, w=weights[~mask])
            else:
                coeffs = np.polyfit(x_masked, data_masked, self.order)
            return np.polyval(coeffs, self.freqs)
        
        except Exception as e:
            log.error(f"Polynomial fitting failed: {e}")
            sys.exit(1)
            
class FitDCT(FitFunc):
    """
    Median filtering class for continuum subtraction 
    """
    def __init__(self, freqs, velwidth, ncoef, dct_type=1, fit_tol=0):
        """
        needs to know the order of the spline and the number of knots
        """
        self.velwidth = velwidth
        self.fit_tol = fit_tol
        self.preped = False
        self.dct_type = dct_type
        self.ncoef = ncoef
        self.freqs = freqs
        self.nchan = freqs.size
        
        fnorm_dict = {
        1: 1 / np.sqrt( 2 * self.nchan),
        2: np.sqrt(2 / (self.nchan - 1)) / 2,
        3: 1 / np.sqrt( 2 * self.nchan),
        4: 1 / np.sqrt( 2 * self.nchan),
        }
        
        self.fnorm = fnorm_dict[dct_type]
        print(self.fnorm)
        
        self.prepare()
            
    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the median filtered data as line emission
        
        x : x values for the fit
        y : values to be fit
        mask : a mask (not implemented really)
        weight : weights
        """
        baseline = FitMedFilter(self.freqs, self.velwidth)
        
        baseline_median = baseline.fit(data, mask=mask, weights=None)
        
        
        dct_data = fftpack.dct(baseline_median, type=self.dct_type)
        sort_idx = np.argsort(np.absolute(dct_data))[:-self.ncoef]
        dct_data[sort_idx] = 0
        
        dct_fit = fftpack.idct(dct_data, type=self.dct_type) * self.fnorm**2
        
        return dct_fit

