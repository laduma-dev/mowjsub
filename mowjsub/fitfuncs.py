from abc import abstractmethod

import numpy as np
from scabha import init_logger
from scipy import fftpack
from scipy.interpolate import make_smoothing_spline, splev, splrep
from scipy.ndimage import median_filter

from mowjsub import utils

from . import BIN
from .exceptions import BadFitError

log = init_logger(BIN.im_plane)


class FitFunc:
    def __init__(
        self,
        freqs,
        order: int = None,
        velwidth: float = None,
        chanwidth: int = None,
        fit_lam: int = None,
        fit_tol: float = 0,
    ):
        """

        Args:
            order (_type_): _description_
            velwidth (_type_): _description_
        """
        self.velwidth = velwidth
        self.fit_tol = fit_tol
        self.freqs = np.asarray(freqs)
        self.nchan = self.freqs.size
        self.order = order
        self.preped = False
        self.chanwidth = chanwidth
        self.fit_lam = fit_lam

    def invalid_point_count(self, data: np.ndarray, mask: np.ndarray):
        """Calculates the number of invalid data points in a spectrum.

        Args:
            data (np.ndarray): 1D spectrum
            mask (np.ndarray): Binary mask
        """
        mask[np.isnan(data)] = True

        return mask, mask.sum()

    def is_fit_possible(self, data: np.ndarray, mask: np.ndarray, raise_exception=True):
        """_summary_

        Args:
            data (np.ndarray): _description_
            mask (np.ndarray): _description_
        """
        mask, invalid = self.invalid_point_count(data, mask)
        nchan = data.size
        valid_fraction = (1 - invalid / nchan) * 100
        if valid_fraction < self.fit_tol:
            if raise_exception:
                raise BadFitError(f"Fraction of valid data points {valid_fraction:.2f} is less than the required tolerance (--cont-fit-tol {self.fit_tol})")
            else:
                return False
        else:
            return mask, invalid

    @abstractmethod
    def fit(self, x, data, mask, weight):
        pass

    def default_prepare(self):
        if self.velwidth:
            self.chanwidth = utils.chans_in_velwidth(self.freqs * 1e6, self.velwidth * 1000)
            log.info(f"Velocity of {self.velwidth} km/s corresponds to {self.chanwidth} channels")
        elif self.chanwidth is None:
            raise RuntimeError("Neither chanwidth or velwitdth are set. Cannot proceed.")

        if self.chanwidth % 2 == 0:
            self.chanwidth += 1

        self.preped = True


class FitBSpline(FitFunc):
    """
    BSpline fitting function based on `splev`, `splrep` in `scipy.interpolate`
    """

    def prepare(self):
        self.default_prepare()

        self.max_spline_order = int(self.nchan / self.chanwidth) + 1
        log.info(f"max spline order: {self.max_spline_order}")

        rs = np.random.SeedSequence()
        self.rng = np.random.default_rng(rs)
        self.preped = True

    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the spline fit and the residuals from the fit

        x : x values for the fit
        data : values to be fit by spline
        weight : weights for fitting the Spline.
            To mask values, set the corresponding weight to zero.
        """

        if not self.preped:
            self.prepare()

        mask, invalid = self.is_fit_possible(data, mask, raise_exception=True)
        nvalid = self.nchan - invalid

        if nvalid < (self.order + 1):
            raise BadFitError("Not enough valid points for spline fit, returning original data.")

        # Mask invalid or zero-weight points
        mask[np.where(np.isnan(data))] = True
        x_masked = self.freqs[~mask]
        data_masked = data[~mask]

        knotind = np.linspace(0, x_masked.size, self.max_spline_order, dtype=int)[1:-1]
        chwid = max(1, (self.nchan // self.max_spline_order) // 8)
        knots_idx = self.rng.integers(-chwid, chwid, size=knotind.shape) + knotind
        knots_idx = np.unique(np.clip(knots_idx, 1, x_masked.size - 1))  # avoid edges

        knot_positions = x_masked[knots_idx]

        if isinstance(weights, np.ndarray):
            splCfs = splrep(
                x_masked,
                data_masked,
                task=-1,
                w=weights[~mask],
                t=knot_positions,
                k=self.order,
            )
        else:
            splCfs = splrep(x_masked, data_masked, task=-1, t=knot_positions, k=self.order)

        return splev(self.freqs, splCfs)


class FitGCVSpline(FitFunc):
    """
    Polynomial fitting function using scipy.interpolate.make_smoothing_spline
    """

    def prepare(self):
        self.preped = True

    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): _description_
            data (np.ndarray): _description_
            weights (np.ndarray): _description_
            mask (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        if not self.preped:
            self.prepare()
        mask, _ = self.is_fit_possible(data, mask, raise_exception=True)

        x_masked = self.freqs[~mask]
        data_masked = data[~mask]

        try:
            if isinstance(weights, np.ndarray):
                smooth_func = make_smoothing_spline(x_masked, data_masked, lam=self.fit_lam, w=weights[~mask])
            else:
                smooth_func = make_smoothing_spline(x_masked, data_masked, lam=self.fit_lam)
        except Exception as e:
            raise BadFitError(f"Polynomial fitting failed: {e}")

        return smooth_func(self.freqs)


class FitMedFilter(FitFunc):
    """
    Median filtering class for continuum subtraction
    """

    def prepare(self):
        self.default_prepare()

    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the median filtered data as line emission

        data (np.ndarray) : values to be fit
        mask (np.ndarray) : a mask (not implemented really)
        weight (np.ndarray) : weights
        """

        if not self.preped:
            self.prepare()

        mask, _ = self.is_fit_possible(data, mask, raise_exception=True)

        if isinstance(weights, np.ndarray):
            # TODO(Sphe)
            pass

        if isinstance(mask, np.ndarray):
            data[mask] = np.nan

        pad_size = int(self.chanwidth / 2)
        padded_data = np.pad(data, pad_size, mode="linear_ramp")
        filtered = np.nanmedian(
            np.lib.stride_tricks.sliding_window_view(padded_data, self.chanwidth),
            axis=1,
        )

        return filtered


class FitMedFilterFast(FitFunc):
    """
    Median filtering class for continuum subtraction
    """

    def prepare(self):
        self.default_prepare()

    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the median filtered data as line emission

        data (np.ndarray) : values to be fit
        mask (np.ndarray) : a mask (not implemented really)
        weight (np.ndarray) : weights
        """

        if not self.preped:
            self.prepare()

        mask = self.is_fit_possible(data, mask, raise_exception=True)[0]

        if isinstance(mask, np.ndarray):
            data[mask] = np.nan

        # Fill NaNs with nearest value for filtering
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data_filled = np.copy(data)
            data_filled[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), data[~nan_mask])
        else:
            data_filled = data

        # Use scipy.ndimage.median_filter for speed
        filtered = median_filter(data_filled, size=self.chanwidth, mode="reflect")

        return filtered


class FitPolynomial(FitFunc):
    """
    Polynomial fitting function using numpy.polyfit
    """

    def prepare(self):
        pass

    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): _description_
            data (np.ndarray): _description_
            weights (np.ndarray): _description_
            mask (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        if not self.preped:
            self.prepare()

        mask, _ = self.is_fit_possible(data, mask, raise_exception=True)

        x_masked = self.freqs[~mask]
        data_masked = data[~mask]

        try:
            if isinstance(weights, np.ndarray):
                coeffs = np.polyfit(x_masked, data_masked, self.order, w=weights[~mask])
            else:
                coeffs = np.polyfit(x_masked, data_masked, self.order)
            return np.polyval(coeffs, self.freqs)

        except Exception as e:
            raise BadFitError(f"Polynomial fitting failed: {e}")


class FitDCT(FitFunc):
    """
    Median filtering class for continuum subtraction
    """

    def prepare(self, dct_type=1):
        self.default_prepare()

        fnorm_dict = {
            1: 1 / np.sqrt(2 * self.nchan),
            2: np.sqrt(2 / (self.nchan - 1)) / 2,
            3: 1 / np.sqrt(2 * self.nchan),
            4: 1 / np.sqrt(2 * self.nchan),
        }

        self.fnorm = fnorm_dict[dct_type]
        self.dct_type = dct_type
        self.preped = True

    def fit(self, data: np.ndarray, mask: np.ndarray, weights: np.ndarray):
        """
        returns the median filtered data as line emission

        x : x values for the fit
        y : values to be fit
        mask : a mask (not implemented really)
        weight : weights
        """

        if not self.preped:
            self.prepare()

        mask, _ = self.is_fit_possible(data, mask, raise_exception=True)

        if isinstance(weights, np.ndarray):
            # TODO(Sphe)
            pass

        baseline = FitMedFilterFast(self.freqs, velwidth=self.velwidth, chanwidth=self.chanwidth)
        baseline_median = baseline.fit(data, mask=mask, weights=None)

        dct_data = fftpack.dct(baseline_median, type=self.dct_type)
        if self.order > 0:
            sort_idx = np.argsort(np.absolute(dct_data))[: -self.order]
            dct_data[sort_idx] = 0
        dct_fit = fftpack.idct(dct_data, type=self.dct_type) * self.fnorm**2

        return dct_fit
