import unittest
import numpy as np
from scabha import init_logger
from contsub.fitfuncs import (
    FitBSpline,
    FitGCVSpline,
    FitMedFilter,
    FitMedFilterFast,
    FitPolynomial,
    #FitDCT,
)
from contsub import utils

log = init_logger("contsub")

class TestFitsFunc(unittest.TestCase):
    def setUp(self):
        self.nchan = nchan = 1000
        xvals = np.linspace(0,2*np.pi, nchan)
        noise = np.random.randn(nchan) * 1.5

        # --- simulate line profile ---
        nterm = 3
        amps = np.random.uniform(0.1, 0.3, nterm)
        shifts = np.random.uniform(0, np.pi/4, nterm)
        omegas = 2*np.pi * np.random.uniform(0.2, 1, nterm)

        line = np.sum([amp*np.sin(omega*xvals+shift) for amp,omega,shift in zip(amps, omegas, shifts)], axis=0)
        line /= amps.sum()

        self.data = data = line + noise

        dfreq = 6500 * 1e-6
        freq0 = 1361
        self.freqs = freqs = freq0 + np.linspace(0, nchan*dfreq, nchan)

        nans = tuple(set(np.random.randint(20,80,10)))
        data[(nans,)] = np.nan
        # ---

        self.mask = mask = np.zeros(nchan, dtype=bool)
        #---- and flags
        mask_start = np.random.randint(10, nchan-10)
        mask_end = mask_start + 20
        mask[mask_start:mask_end] = True

        mask_start = np.random.randint(10, nchan-10)
        mask_end = mask_start + 40
        mask[mask_start:mask_end] = True
        #----

        self.velwidth = 300
        self.chanwidth = utils.chans_in_velwidth(freqs*1e6, self.velwidth*1e3)
    
    def test_median_filter(self):
        
        baseline_func = FitMedFilter(self.freqs, velwidth=self.velwidth)
        baseline = baseline_func.fit(self.data, mask=self.mask, weights=None)
        
        assert baseline.shape == self.data.shape
    
    def test_median_filter_fast(self):
        
        baseline_func = FitMedFilterFast(self.freqs, velwidth=self.velwidth)
        baseline_vel = baseline_func.fit(self.data, mask=self.mask, weights=None)
        assert baseline_vel.shape == self.data.shape
        
        # test if chanwidth gives same result as velwidth
        baseline_func = FitMedFilterFast(self.freqs, chanwidth=self.chanwidth)
        baseline_chan = baseline_func.fit(self.data, mask=self.mask, weights=None)
        
        assert np.allclose(baseline_vel, baseline_chan, atol=1e-6)

    def test_polynomial(self):
        
        baseline_func = FitPolynomial(self.freqs, order=3)
        baseline = baseline_func.fit(self.data, mask=self.mask, weights=None)
        
        assert baseline.shape == self.data.shape

    def test_b_spline(self):
        
        baseline_func = FitBSpline(self.freqs, order=3, velwidth=self.velwidth)
        baseline_vel = baseline_func.fit(self.data, mask=self.mask, weights=None)
        
        # test if chanwidth gives same result as velwidth
        baseline_func = FitBSpline(self.freqs, order=3, chanwidth=self.chanwidth)
        baseline_chan = baseline_func.fit(self.data, mask=self.mask, weights=None)
        
        assert baseline_chan.shape == self.data.shape
        baseline_chan_std = baseline_chan.std()
        
        perr = np.abs(baseline_vel.std() - baseline_chan_std) / baseline_chan_std
        # tolerate a 5% error because the knots are chosen using a random generator
        assert perr < 5/100
        
    def test_gcv_spline(self):
        
        baseline_func = FitGCVSpline(self.freqs)
        baseline = baseline_func.fit(self.data, mask=self.mask, weights=None)
        
        assert baseline.shape == self.data.shape
