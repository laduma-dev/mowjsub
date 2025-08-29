import pytest
from contsub.image_plane import ContSub
from contsub.fitfuncs import FitBSpline
import numpy as np


# Test the ContSub class
def test_contsub_initialization():
    # Create a ContSub instance with a mock FitFunc
    fit_func = FitBSpline(order=3, velWidth=1000, randomState=None, seq=None)
    contsub = ContSub(fit_func=fit_func, nomask=True, fit_tol=0)

    # Check if the instance is created correctly
    assert isinstance(contsub, ContSub)
    assert contsub.fit_func == fit_func
    
    
def test_contsub_fitContinuum():
    # Create a ContSub instance with a mock FitFunc
    fit_func = FitBSpline(order=3, velWidth=200, randomState=None, seq=None)
    contsub = ContSub(fit_func=fit_func, nomask=True, fit_tol=0)

    # Mock data for xspec and cube
    nchan = 128
    npix = 10
    xspec = np.linspace(1411, 1412, nchan)  # Mock spectral coordinates
    cube = np.random.randn(npix, npix, nchan) * 1e-3  # Mock data cube
    mask = None  # No mask

    # Call the fitContinuum method
    cont_model = contsub.fitContinuum(xspec, cube, mask)

    # Check if the output is as expected (mocked for simplicity)
    assert cont_model is not None
    assert cont_model.shape == cube.shape
    
    
def test_contsub_fitContinuum_with_mask():
    # Create a ContSub instance with a mock FitFunc
    fit_func = FitBSpline(order=3, velWidth=200, randomState=None, seq=None)
    contsub = ContSub(fit_func=fit_func, nomask=False, fit_tol=0)

    # Mock data for xspec and cube
    nchan = 128
    npix = 10
    xspec = np.linspace(1411, 1412, nchan)  # Mock spectral coordinates
    cube = np.random.randn(npix, npix, nchan) * 1e-3  # Mock data cube
    mask = np.random.choice([True, False], size=(npix, npix, nchan))  # Random mask

    # Call the fitContinuum method
    cont_model = contsub.fitContinuum(xspec, cube, mask)

    # Check if the output is as expected (mocked for simplicity)
    assert cont_model is not None
    assert cont_model.shape == cube.shape

