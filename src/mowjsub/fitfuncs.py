import logging
from abc import abstractmethod

import numpy as np
from scipy import fftpack
from scipy.interpolate import make_smoothing_spline, splev, splrep
from scipy.ndimage import median_filter

from mowjsub import utils

from . import LOGGER
from .exceptions import BadFitError

log = logging.getLogger(LOGGER)


class FitFunc:
    def __init__(
        self,
        freqs,
        order: int = None,
        velwidth: float = None,
        chanwidth: int = None,
        fit_lam: int = None,
        fit_tol: float = 0,
        seed=None,
    ):
        """

        Args:
            order (_type_): _description_
            velwidth (_type_): _description_
            seed: Seed for fitters that place knots at random offsets. None
                (the default) draws fresh entropy, so repeated fits of the same
                spectrum differ slightly. Pass an integer to make a run
                reproducible.
        """
        self.velwidth = velwidth
        self.fit_tol = fit_tol
        self.freqs = np.asarray(freqs)
        self.nchan = self.freqs.size
        self.order = order
        self.preped = False
        self.chanwidth = chanwidth
        self.fit_lam = fit_lam
        self.seed = seed

    def invalid_point_count(self, data: np.ndarray, mask: np.ndarray):
        """Calculates the number of invalid data points in a spectrum.

        Args:
            data (np.ndarray): 1D spectrum
            mask (np.ndarray): Binary mask

        Returns:
            tuple: A new mask with NaN data points flagged, and the invalid point count.
        """
        mask = mask | np.isnan(data)

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

    def ascending(self, x, data, weights):
        """Present a fitter's input in increasing-x order.

        scipy's spline fitters require strictly increasing x, and a FITS
        spectral axis descends whenever ``CDELT`` is negative -- which is
        *always* the case for a cube whose axis is a velocity or a wavelength,
        since both run opposite to frequency. Handed such a grid, ``splrep``
        raised a bare ``ValueError: Error on input data`` that killed the run,
        and ``make_smoothing_spline`` failed per spectrum, which
        ``ContSub.fitContinuum`` turned into an all-NaN cube and no error at all.

        Only the *fitting* input needs reordering. ``splev`` and a smoothing
        spline both evaluate at whatever order they are handed, so the fit is
        still evaluated on ``self.freqs`` and lines up with the data channel for
        channel. A spline is invariant under reversing x and y together, so this
        changes no result that already worked.

        Args:
            x (np.ndarray): Frequency grid, masked points already removed.
            data (np.ndarray): Spectrum, matching ``x``.
            weights (np.ndarray|None): Weights, matching ``x``, or None.

        Returns:
            tuple: ``(x, data, weights)``, reversed together if x descended.
        """
        if x.size > 1 and x[0] > x[-1]:
            return x[::-1], data[::-1], weights[::-1] if isinstance(weights, np.ndarray) else weights

        return x, data, weights

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

        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))
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
        x_masked, data_masked, weights_masked = self.ascending(
            self.freqs[~mask],
            data[~mask],
            weights[~mask] if isinstance(weights, np.ndarray) else weights,
        )

        knotind = np.linspace(0, x_masked.size, self.max_spline_order, dtype=int)[1:-1]
        chwid = max(1, (self.nchan // self.max_spline_order) // 8)
        knots_idx = self.rng.integers(-chwid, chwid, size=knotind.shape) + knotind

        # FITPACK imposes two limits on the interior knots, and neither was
        # enforced. They must lie *strictly* inside the data range, so the last
        # usable index is `size - 2`; the bound here was `size - 1`, which is
        # x[-1] itself. And there can be at most `m - k - 1` of them for m
        # points of order k. Both are reachable through --vel-width, which is a
        # physical width and so asks for one segment per channel on a coarsely
        # channelised cube: `max_spline_order` then approaches the channel count,
        # packing the jittered knots right up against the end of the spectrum.
        #
        # Neither failed usefully. FITPACK rejects the fit from inside Fortran
        # as a bare `TypeError: An error occurred` or
        # `ValueError: Error on input data`, naming no parameter, and since the
        # exception escapes `ContSub.fitContinuum` it took the whole run with it
        # rather than marking the spectrum bad.
        max_knots = x_masked.size - self.order - 1
        if max_knots <= 0:
            knot_positions = []
        else:
            knots_idx = np.unique(np.clip(knots_idx, 1, x_masked.size - 2))
            if knots_idx.size > max_knots:
                # Thin evenly rather than truncate, so the knots still span the
                # band instead of crowding into its low-frequency end.
                knots_idx = knots_idx[np.linspace(0, knots_idx.size - 1, max_knots, dtype=int)]

            knot_positions = x_masked[knots_idx]

        if isinstance(weights, np.ndarray):
            splCfs = splrep(
                x_masked,
                data_masked,
                task=-1,
                w=weights_masked,
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

        x_masked, data_masked, weights_masked = self.ascending(
            self.freqs[~mask],
            data[~mask],
            weights[~mask] if isinstance(weights, np.ndarray) else weights,
        )

        try:
            if isinstance(weights, np.ndarray):
                smooth_func = make_smoothing_spline(x_masked, data_masked, lam=self.fit_lam, w=weights_masked)
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
            data = np.where(mask, np.nan, data)

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

        # is_fit_possible folds NaN data points into the mask, so masked and
        # non-finite points are the same set here.
        nan_mask = mask

        # Fill NaNs with nearest value for filtering
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
