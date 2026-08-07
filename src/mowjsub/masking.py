import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import ndimage
from scipy.signal import convolve

from mowjsub import LOGGER

log = logging.getLogger(LOGGER)


def filter_components(det, min_channels=1, sky_element=None, structure=None):
    """Keep only detection components big enough in frequency and on the sky.

    The sky test is an erosion by ``sky_element`` rather than a pixel count:
    counting is orientation-blind, so an elongated streak with as many pixels as
    a beam would pass it. Eroding asks whether the component's sky projection
    actually *contains* a region of that shape, which is what makes a detection
    credible -- nothing real is smaller than the beam.

    Args:
        det (np.ndarray): Boolean ``(ra, dec, spectral)``, True where detected.
        min_channels (int): Channels a component must span.
        sky_element (np.ndarray|None): Beam-shaped structuring element from
            :func:`~mowjsub.sources.beam_element`. ``None`` skips the sky test.
        structure (np.ndarray|None): Connectivity for labelling. Defaults to
            the 18-neighbour structure the mask growth uses, so a component
            means the same thing to both.

    Returns:
        np.ndarray: ``det`` with the components below either bound removed.
    """
    if min_channels <= 1 and sky_element is None:
        return det
    if structure is None:
        structure = ndimage.generate_binary_structure(det.ndim, 2)

    labels, found = ndimage.label(det, structure=structure)
    if not found:
        return det

    # A component spanning C channels and covering an element of E pixels holds
    # at least max(C, E) voxels, so this prunes almost everything before the
    # per-component work, which needs a slice each.
    floor = max(min_channels, 0 if sky_element is None else int(sky_element.sum()))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    candidates = np.flatnonzero(sizes >= floor)
    if candidates.size == 0:
        return np.zeros_like(det)

    boxes = ndimage.find_objects(labels)
    keep = []
    for label in candidates:
        box = boxes[label - 1]
        if box is None:
            continue
        # Cheap necessary condition first: the bounding box bounds the extent.
        if (box[2].stop - box[2].start) < min_channels:
            continue
        component = labels[box] == label
        if component.any(axis=(0, 1)).sum() < min_channels:
            continue
        if sky_element is not None:
            sky = component.any(axis=2)
            # border_value=0: an element hanging off the edge of the bounding
            # box does not count as contained.
            if not ndimage.binary_erosion(sky, structure=sky_element, border_value=0).any():
                continue
        keep.append(label)

    if not keep:
        return np.zeros_like(det)

    return np.isin(labels, keep)


class Mask:
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

    def __init__(self, n, sm_kernel=None, dilation=0, method="rms", min_channels=1, sky_element=None):
        """
        has to define the multiple of sigma for clipping and the method for calculating the sigma

        n : multiple of sigma for clipping
        method : 'rms' or 'mad' for calculating the rms
        min_channels : smallest spectral extent, in channels, a detection
            component must span to survive into the mask.
        sky_element : beam-shaped structuring element a component's sky
            projection must contain, from ``sources.beam_element``. Stated as a
            shape rather than an area because a pixel count is
            orientation-blind, and sized in beams rather than pixels because a
            pixel means something different on every cube.

        The defaults -- 1 channel, no element -- filter nothing, reproducing
        the mask as it behaved before these parameters existed.

        Why two axes rather than a voxel count. ``createMask`` grows every
        detection by two 18-neighbour passes, which is what catches the line
        wings that sit below any threshold their peak clears -- worth up to 22
        points of recovered line flux -- but it grows noise identically. A cube's
        noise is correlated across the beam and independent between channels, so
        noise components are spatially extended but spectrally thin: measured on
        beam-correlated noise, they span a median of 1 channel and never more
        than 2, while covering a median of 3 sightlines. A real line spans both.
        Requiring 2 channels and a half-beam element left 0.7% of noise
        components alive, while a 3-sigma resolved line came through at 89
        voxels across 3 channels.

        A voxel count alone does not separate them, because noise and signal
        both occupy about a beam: on beam-correlated noise a cut of 2 voxels
        moves the excluded fraction only from 8.2% to 6.6%. (An earlier
        measurement of 19.1% -> 0.5% was taken on *white* noise, which no real
        cube has, and it does not transfer -- on the chi-oph cube that cut
        changed the products by 0.01%.)
        """
        self.n = n
        self.dilate = dilation
        self.min_channels = min_channels
        self.sky_element = sky_element
        if sm_kernel is None:
            self.sm = None
        else:
            sm_kernel = np.array(sm_kernel)
            if len(sm_kernel.shape) == 1:
                self.sm = sm_kernel[:, None, None]
            else:
                self.sm = sm_kernel
        if method == "rms":
            self.function = self.__rms()
        elif method == "mad":
            self.function = self.__mad()

    def createMask(self, data):
        """
        calculate a mask from the given data
        """
        sm_data = self.__smooth(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sigma = self.function(sm_data)[..., np.newaxis]
        mask = np.abs(sm_data) < self.n * sigma

        struct_dil = ndimage.generate_binary_structure(len(data.shape), 1)
        struct_erd = ndimage.generate_binary_structure(len(data.shape), 2)

        # Drop detections too small in sky or in frequency before the growth
        # below reaches them. Labelled with the same structure the growth uses,
        # so "connected" means the same thing to both.
        if self.min_channels > 1 or self.sky_element is not None:
            det = ~mask
            det = filter_components(det, self.min_channels, self.sky_element, struct_erd)
            mask = ~det

        for i in range(self.dilate):
            mask = ndimage.binary_dilation(mask, structure=struct_dil, border_value=1).astype(mask.dtype)

        for i in range(self.dilate + 2):
            mask = ndimage.binary_erosion(mask, structure=struct_erd, border_value=1).astype(mask.dtype)

        return mask

    def __smooth(self, data):
        if self.sm is None:
            return data
        else:
            sm_data = convolve(data, self.sm, mode="same")
            return sm_data

    def __rms(self):
        return lambda x: np.sqrt(np.nanmean(np.square(x), axis=2))

    def __mad(self):
        return lambda x: np.nanmedian(np.abs(np.nanmean(x) - x), axis=2)


class ChanSigmaClip(ClipMethod):
    """
    simple sigma clipping class
    """

    def __init__(self, n, method="rms"):
        """
        has to define the multiple of sigma for clipping and the method for calculating the sigma

        n : multiple of sigma for clipping
        method : 'rms' or 'mad' for calculating the rms
        """
        self.n = n
        if method == "rms":
            self.function = self.__rms()
        elif method == "mad":
            self.function = self.__mad()

    def createMask(self, data):
        """
        calculate a mask from the given data
        """
        sigma = self.function(data)[:, None, None]
        return np.abs(data) < self.n * sigma

    def __rms(self):
        return lambda x: np.sqrt(np.nanmean(np.square(x), axis=(0, 1)))

    def __mad(self):
        return lambda x: np.nanmedian(np.abs(np.nanmean(x) - x), axis=(0, 1))
