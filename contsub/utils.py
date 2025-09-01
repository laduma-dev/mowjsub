from xarrayfits import xds_from_fits
import xarray as xr
from astropy.wcs import WCS
from contsub.image_plane import ContSub
from contsub.masking import PixSigmaClip, Mask
from contsub.fitfuncs import FitBSpline
from contsub import BIN
from typing import Dict
from scabha import init_logger
import astropy.units as u
import astropy.io.fits as fitsio
from scabha.basetypes import File
import numpy as np
import datetime
log = init_logger(BIN.im_plane)


def get_automask(xspec, cube, sigma_clip=5, order=3, segments=400):
    """
    Generate a binary mask by sigma-thresholding the input cube

    Args:
        xspec (Array): Spectral coordinates
        cube (Array): Data cdube
        sigma_clip (int, optional): _description_. Defaults to 5.
        order (int, optional): _description_. Defaults to 3.
        segments (int, optional): Length of spline segment in km/s. Defaults to 400.

    Returns:
        Array : Binary mask (False is masked, True is not)
    """

    log.info("Creating binary mask as requested")
    fitfunc = FitBSpline(order, segments)
    contsub = ContSub(fitfunc, nomask=True, fit_tol=60)
    cont_model = contsub.fitContinuum(xspec, cube, mask=None)
    
    clip = PixSigmaClip(sigma_clip)
        
    mask = Mask(clip).getMask(cube - cont_model)
    log.info("Mask created sucessfully")
    
    return mask


def zds_from_fits(fname, chunks=None, rest_freq=None, hdu_idx=0, add_freqs=False):
    """ Creates Zarr store from a FITS file. The resulting array has 
    dimensions = RA, DEC, SPECTRAL[, STOKES]

    Args:
        fname (str|path): FITS file_
        chunks (dict, optional): xarray chunk object. Defaults to {1: 25, 2:25}.

    Raises:
        RuntimeError: Input FITS file doesn't have a spectral axis
        FileNotFoundError: Input FITS file not found

    Returns:
        Zarr: Zarr array (persistant store, mode=w)
    """
    chunks = chunks or dict(ra=64,dec=None, spectral=None)
    fds = xds_from_fits(fname, hdus=hdu_idx)[0]
    
    header = fds.hdu.header
    if rest_freq:
        header["RESTFREQ"] = rest_freq * 1e6 # set it to Hz
    wcs = WCS(header, naxis="spectral stokes".split())
    

    axis_names = [header["CTYPE1"], header["CTYPE2"]] + wcs.axis_type_names
    if not wcs.has_spectral:
        raise RuntimeError("Input FITS file does not have a spectral axis")

    fds_xyz = fds.hdu.transpose(*axis_names)

    new_names = ["ra", "dec", "spectral"]
    if len(axis_names) == 4:
        new_names.append("stokes")

    coords = dict([(a,fds.hdu[b].values) for a,b in zip(new_names,axis_names)])
    if add_freqs:
        data_vars = {
            "DATA": (new_names, fds_xyz.data),
            "FREQS": (("spectral",), FitsHeader(header).retFreq()), 
        }
    else:
        data_vars = {
            "DATA" : (new_names, fds_xyz.data),
        }
    ds = xr.Dataset(
        data_vars,
        coords = coords,
        attrs = dict(
            info =f"Temporary copy of data from FITS file: {fname}",
            header = fitsio.Header(header),
                    ),
    )

    return ds.chunk(chunks)

def subtract_fits(data_file: File, model_file: File, chunks: Dict):
    """ Returns the residual of two FITS files as a FitsPrimaryHDU object

    Args:
        data_file (File): FITS file of the data
        model_file (File): FITS file of model data
        chunks (Dict): How to chunk the data

    Returns:
        FitsPrimaryHDU
    """
    
    data_ds = xds_from_fits(data_file, chunks=chunks)[0]
    model_ds = xds_from_fits(model_file, chunks=chunks)[0]
    residual_ds = data_ds.hdu.data - model_ds.hdu.data
    
    out_ds = data_ds.assign(hdu=(
        data_ds.hdu.dims,
        residual_ds, 
        data_ds.hdu.attrs),
    )
    header = fitsio.Header(data_ds.hdu.header)
    return fitsio.PrimaryHDU(out_ds.hdu.data, header=header)


class FitsHeader():
    def __init__(self, header: Dict):
        self._header = header.copy()
        
    def retFreq(self):
        """
        Extract the part of the cube name that will be used in the name of
        the averaged cube

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            header object from the fits file

        Returns
        -------
        frequency
            a 1D numpy array of channel frequencies in MHz  
        """
        
        if not ('TIMESYS' in self._header):
            self._header['TIMESYS'] = 'utc'
        elif self._header['TIMESYS'] != 'utc':
            self._header['TIMESYS'] = 'utc'
        freqDim = self._header['NAXIS3']
        wcs3d=WCS(self._header)
        try:
            wcsfreq = wcs3d.spectral
        except:
            wcsfreq = wcs3d.sub(['spectral'])   
        return np.around(wcsfreq.pixel_to_world(np.arange(0,freqDim)).to(u.MHz).value, decimals = 7)
        
    def getAppendHeader(self, nchan):
        return self.spectralSplitHeader(nchan, orig = 'append_fits')
    
    def getTableHeader(self, nchan):
        self._header['NAXIS2'] = nchan
        self._header['NCHAN'] = nchan
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = 'A. Kazemi-Moridani (table_header)'
        return self._header
    
    def getCombineHeader(self, dimx, dimy):
        self._header['NAXIS1'] = int(dimx)
        self._header['NAXIS2'] = int(dimy)
        # self._header['CRPIX1'] = xcen 
        # self._header['CRPIX2'] = ycen
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = 'A. Kazemi-Moridani (combine_spatial)'
        return self._header
    
    
    def getPrimeHeader(self, nchan, ydim, xdim, mask = False, orig = 'prime_header'):
        self._header['NAXIS1'] = int(xdim)
        self._header['NAXIS2'] = int(ydim)
        self._header['NAXIS3'] = int(nchan)
        if mask:
            self._header['BITPIX'] = 8
            if orig == 'prime_header':
                orig = 'mask_header'
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = f'A. Kazemi-Moridani ({orig})'
        return self._header
    
    def spectralSplitHeader(self, nchan, sfreq = None, orig = 'spectral_split'):
        self._header['NAXIS3'] = nchan
        if sfreq != None:
            self._header['CRVAL3'] = sfreq
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = f'A. Kazemi-Moridani ({orig})'
        return self._header
    
    def spatialSplitHeader(self, xlims, ylims, chans = None):
        xdn, xup = xlims
        ydn, yup = ylims
        freq = self.retFreq()
        if chans != None:
            chdn, chup = chans
            self._header['NAXIS3'] = chup - chdn #make sure this is correct it was 'chup - chdn + 1' before
            self._header['CRVAL3'] = freq[chdn]*1e6
        xor = self._header['CRPIX1'] 
        yor = self._header['CRPIX2'] 
        self._header['CRPIX1'] = xor - xdn
        self._header['CRPIX2'] = yor - ydn
        self._header['NAXIS1'] = xup-xdn
        self._header['NAXIS2'] = yup-ydn
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = 'A. Kazemi-Moridani (spatial_split)'
        return self._header
