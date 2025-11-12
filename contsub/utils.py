import xarray as xr
from astropy.wcs import WCS
from contsub.masking import PixSigmaClip, Mask
from contsub.image_plane import ContSub
from contsub import BIN
from typing import Dict
from scabha import init_logger
from astropy import units
from astropy.io import fits
from scabha.basetypes import File
import numpy as np
import datetime
from daskms import xds_from_ms, xds_from_table
from xarrayfits import xds_from_fits
import dask.array as da
from tqdm.dask import TqdmCallback
import warnings

warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
warnings.filterwarnings("ignore", message=".*Consolidated metadata is currently not part.*")

log = init_logger(BIN.im_plane)


def get_automask(cube, fitfunc, sigma_clip):
    """
    Generate a binary mask by sigma-thresholding the input cube

    Args:
        xspec (Array): Spectral coordinates
        cube (Array): Data cdube
        sigma_clip(float): Sigma clip level

    Returns:
        Array : Binary mask (False is masked, True is not)
    """

    log.info("Creating binary mask as requested")
    contsub = ContSub(fitfunc)
    cont_model = contsub.fitContinuum(cube, mask=None)
    
    clip = PixSigmaClip(sigma_clip)
        
    mask = Mask(clip).getMask(cube - cont_model)
    log.info("Mask created sucessfully")
    
    return ~mask

def chans_in_velwidth(freqs:np.ndarray, velwidth:float):
    """
    Calculates the number of channels in given velocity width

    Args:
        freqs (np.ndarray): Frequency grid in Hz.
        velwidth (float): Velocity width in m/s
    """
    speed_c = 2.998e8
    df_low = np.partition(freqs, 2)[:2]
    df_high = np.partition(freqs, 2)[-2:]
    
    dv_high = np.abs(np.diff(df_low) / np.mean(df_low)) * speed_c
    dv_low = np.abs(np.diff(df_high) / np.mean(df_high)) * speed_c
    dv = np.mean([dv_low, dv_high])
    
    return int(velwidth / dv)


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

    n_axes = fds.hdu.data.ndim

    if n_axes == 4:
        new_names = ["ra", "dec", "spectral", "stokes"]
        fds_xyz = fds.hdu.transpose(*axis_names[:4])
    elif n_axes == 3:
        new_names = ["ra", "dec", "spectral"]
        fds_xyz = fds.hdu.transpose(*axis_names[:3])
    else:
        raise RuntimeError(f"Unexpected number of data axes ({n_axes}) in FITS file")
    
    #if len(axis_names) == 4:
        #new_names.append("stokes")

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
            header = header,
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
    header = fits.Header(data_ds.hdu.header)
    return fits.PrimaryHDU(out_ds.hdu.data, header=header)


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
        return np.around(wcsfreq.pixel_to_world(np.arange(0,freqDim)).to(units.MHz).value, decimals = 7)
        
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
    
def get_ds_from_msdsl(ms_dsl, field_id=0, data_desc_id=0):
    found_ds = False
    for ds in ms_dsl:
        if ds.FIELD_ID == field_id and ds.DATA_DESC_ID == data_desc_id:
            found_ds = True
            break
    if found_ds:
        return ds
    else:
        raise ValueError("Dataset with FIELD_ID=1 and DATA_DESC_ID=1 not found in the MS.")
    
def ms_to_xarray_dataset(ms_path, spw_id:int, field_id:int, chunks:int,
                        outchunks = {'time': 64, 'baseline': 64}, save_to_zarr=False):
    
    """ Creates Zarr store from a input MS. The resulting array has 
    dimensions = time, basline, SPECTRAL, corr

    Args:
        ms path (str|path): MS file_
        spw_id (int): Spectral window ID
        field_id (int): Field ID
        chunks (int): How to chunk the data
        outchunks (dict, optional): xarray chunk object. Defaults to {'time': 64, 'baseline': 64}.
        save_to_zarr (bool, optional): Save the output to Zarr. Defaults to False.  
    Returns:
        Zarr: Zarr array (persistant store, mode=w)
    """
    ms_dsl = xds_from_ms(
        ms_path, 
        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={"row": chunks } 
    )
    
    spw_table = xds_from_table(f"{ms_path}::SPECTRAL_WINDOW")[0]
    field_table = xds_from_table(f"{ms_path}::FIELD")[0]
    antenna_table = xds_from_table(f"{ms_path}::ANTENNA")[0]

    frequencies = spw_table.CHAN_FREQ.data[spw_id] 
    channel_width = spw_table.CHAN_WIDTH.data[spw_id][0]  
    ref_frequency = spw_table.REF_FREQUENCY.data[spw_id]

    phase_center = field_table.PHASE_DIR.data[field_id].compute()[0] 

    antenna_names = [name.strip() for name in antenna_table.NAME.data.compute()]
    ds = get_ds_from_msdsl(ms_dsl, field_id=field_id, data_desc_id=spw_id)
    
    times = ds.TIME.data 
    visibilities = ds.DATA.data
    flags = ds.FLAG.data
    weights = ds.WEIGHT_SPECTRUM.data 
    uvw = ds.UVW.data
    
    nant = antenna_table.NAME.size
    nbl = nant*(nant-1) // 2
    nrow, nchan, ncorr = ds.DATA.shape
    ntimes = nrow // nbl

    unique_times = np.unique(times)

    reshaped_vis = da.reshape(visibilities, (ntimes, nbl, nchan, ncorr))
    reshaped_flags = da.reshape(flags, (ntimes, nbl, nchan, ncorr))
    reshaped_weights = da.reshape(weights, (ntimes, nbl, nchan, ncorr))
    
    if ncorr == 2:  
        corr_labels = ['XX', 'YY'][:ncorr]
    else:
        corr_labels = ['XX', 'XY', 'YX', 'YY'][:ncorr]

    dataset = xr.Dataset(
        {   #TO-DO: reshape the data for all 
            'VIS': ([ 'time', 'baseline' , 'spectral', 'corr'], reshaped_vis), #time here will be time indicies
            'FLAG': ([ 'time', 'baseline', 'spectral', 'corr'], reshaped_flags),
            'WEIGHT': ([ 'time', 'baseline', 'spectral', 'corr'], reshaped_weights),
            'UVW': (('time', 'baseline', 'uvw'), uvw.reshape(ntimes, nbl, 3)),
        },
        coords={
            'FREQ': frequencies,  
            'CORR':corr_labels, 
            'TIME': unique_times,
            'BASELINE': np.arange(nbl),
        },
        attrs={
            
            'ref_freq': float(ref_frequency),
            'channel_width': float(channel_width),
            'phase_center_ra': float(phase_center[0]),  # radians
            'phase_center_dec': float(phase_center[1]),  # radians
            'phase_center_ra_deg': float(np.degrees(phase_center[0])),  # degrees
            'phase_center_dec_deg': float(np.degrees(phase_center[1])),  # degrees
            'antenna_names': antenna_names,
            'nant': nant,
            'DATA_DESC_ID': spw_id,
            'FIELD_ID': field_id
        })
    dataset = dataset.chunk(outchunks)
    if not save_to_zarr:
        return dataset
    else:
        outpath = f'tmp.zarr'
        write_to_zarr = dataset.to_zarr(outpath, mode='w', compute=False)

    with TqdmCallback(desc="Writing to Zarr"):
        da.compute(write_to_zarr)

    return outpath