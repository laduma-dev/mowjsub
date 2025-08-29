import numpy as np
from scipy.signal import convolve
from scipy import ndimage
from scabha import init_logger
from abc import ABC, abstractmethod
from contsub import BIN
import warnings

log = init_logger(BIN.im_plane)

class Mask():
    """
    mask class creates a mask using a specific masking method
    """
    def __init__(self, method):
        """
        method should be defined when creating a Mask object
        Method should be built on the ClipMethod class
        """
        self.method = method
        
    def getMask(self, data):
        """
        calculates the mask given the data
        """
        return self.method.createMask(data)

        
class ClipMethod(ABC):
    """
    Abstract class for different methods of making masks
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def createMask(self, data):
        pass
    
class PixSigmaClip(ClipMethod):
    """
    simple sigma clipping class
    """
    def __init__(self, n, sm_kernel = None, dilation = 0, method = 'rms'):
        """
        has to define the multiple of sigma for clipping and the method for calculating the sigma
        
        n : multiple of sigma for clipping
        method : 'rms' or 'mad' for calculating the rms
        """
        self.n = n
        self.dilate = dilation
        if sm_kernel is None:
            self.sm = None
        else:
            sm_kernel = np.array(sm_kernel)
            if len(sm_kernel.shape) == 1:
                self.sm = sm_kernel[:, None, None]
            else:
                self.sm = sm_kernel
        if method == 'rms':
            self.function = self.__rms()
        elif method == 'mad':
            self.function = self.__mad()
        
    def createMask(self, data):
        """
        calculate a mask from the given data 
        """
        sm_data = self.__smooth(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) 
            sigma = self.function(sm_data)[...,np.newaxis]
        mask = np.abs(sm_data) < self.n*sigma
        
        struct_dil = ndimage.generate_binary_structure(len(data.shape), 1)
        struct_erd = ndimage.generate_binary_structure(len(data.shape), 2)
        
        for i in range(self.dilate):
            mask = ndimage.binary_dilation(mask, structure=struct_dil,
                                        border_value=1).astype(mask.dtype)
            
        for i in range(self.dilate+2):
            mask = ndimage.binary_erosion(mask, structure=struct_erd,
                                        border_value=1).astype(mask.dtype)
            
        return mask
    
    def __smooth(self, data):
        if self.sm is None:
            return data
        else:
            sm_data = convolve(data, self.sm, mode = 'same')
            return sm_data
    
    def __rms(self):
        return lambda x: np.sqrt(np.nanmean(np.square(x), axis = 2))
    
    def __mad(self):
        return lambda x: np.nanmedian(np.abs(np.nanmean(x)-x), axis = 2)
        
class ChanSigmaClip(ClipMethod):
    """
    simple sigma clipping class
    """
    def __init__(self, n, method = 'rms'):
        """
        has to define the multiple of sigma for clipping and the method for calculating the sigma
        
        n : multiple of sigma for clipping
        method : 'rms' or 'mad' for calculating the rms
        """
        self.n = n
        if method == 'rms':
            self.function = self.__rms()
        elif method == 'mad':
            self.function = self.__mad()
        
    def createMask(self, data):
        """
        calculate a mask from the given data 
        """
        sigma = self.function(data)[:,None,None]
        return np.abs(data) < self.n*sigma
    
    def __rms(self):
        return lambda x: np.sqrt(np.nanmean(np.square(x), axis = (0,1)))
    
    def __mad(self):
        return lambda x: np.nanmedian(np.abs(np.nanmean(x)-x), axis = (0,1))
