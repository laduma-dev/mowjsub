"""Catalogue reading, footprint building, and the masked-fraction guard."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from mowjsub.sources import footprint_from_catalogue, footprint_from_mask, read_catalogue
from mowjsub.utils import cap_masked_fraction, get_automask


def _header(nra=32, ndec=32, ra0=150.0, dec0=-30.0, cdelt=1e-3):
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"], hdr["NAXIS2"] = nra, ndec
    hdr["CTYPE1"], hdr["CRVAL1"], hdr["CDELT1"], hdr["CRPIX1"] = "RA---SIN", ra0, -cdelt, nra // 2
    hdr["CTYPE2"], hdr["CRVAL2"], hdr["CDELT2"], hdr["CRPIX2"] = "DEC--SIN", dec0, cdelt, ndec // 2
    return hdr


class TestReadCatalogue(unittest.TestCase):
    def test_plain_ra_dec_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cat.txt"
            path.write_text("# ra dec\n150.0 -30.0\n150.1 -30.1\n")
            ra, dec, maj = read_catalogue(path)

        np.testing.assert_allclose(ra, [150.0, 150.1])
        np.testing.assert_allclose(dec, [-30.0, -30.1])
        assert np.isnan(maj).all(), "a two-column list states no extent"

    def test_plain_list_with_major_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cat.txt"
            path.write_text("150.0 -30.0 0.002\n")
            _, _, maj = read_catalogue(path)

        np.testing.assert_allclose(maj, [0.002])

    def test_a_bdsf_style_table_is_read_by_column_name(self):
        from astropy.table import Table

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cat.fits"
            Table({"RA": [150.0], "DEC": [-30.0], "Maj": [0.003], "Total_flux": [1.0]}).write(path)
            ra, dec, maj = read_catalogue(path)

        np.testing.assert_allclose(ra, [150.0])
        np.testing.assert_allclose(maj, [0.003])

    def test_a_table_without_positions_says_which_columns_it_had(self):
        from astropy.table import Table

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cat.fits"
            Table({"Total_flux": [1.0], "Peak_flux": [1.0]}).write(path)
            with self.assertRaises(ValueError) as raised:
                read_catalogue(path)

        assert "Total_flux" in str(raised.exception)


class TestFootprint(unittest.TestCase):
    def test_a_source_at_the_reference_pixel_lands_there(self):
        hdr = _header()
        fp = footprint_from_catalogue(hdr, [150.0], [-30.0], radius_pix=2.0, shape=(32, 32))

        assert fp[16, 16], "the source is at CRPIX, which is pixel (16, 16) zero-based"
        assert not fp[0, 0]
        # A disc of radius 2 covers ~13 pixels, not the whole image.
        assert 5 <= fp.sum() <= 25, fp.sum()

    def test_a_source_outside_the_image_is_dropped_not_wrapped(self):
        hdr = _header()
        fp = footprint_from_catalogue(hdr, [160.0], [-10.0], radius_pix=2.0, shape=(32, 32))

        assert not fp.any(), "a source off the image must not alias back onto it"

    def test_a_catalogued_extent_beats_the_default_radius(self):
        hdr = _header(cdelt=1e-3)
        small = footprint_from_catalogue(hdr, [150.0], [-30.0], maj_deg=[np.nan], radius_pix=2.0, shape=(32, 32))
        # 0.005 deg at 1e-3 deg/pix is a 5 pixel radius.
        big = footprint_from_catalogue(hdr, [150.0], [-30.0], maj_deg=[0.005], radius_pix=2.0, shape=(32, 32))

        assert big.sum() > small.sum()

    def test_mask_shape_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.fits"
            fits.PrimaryHDU(np.ones((8, 8), dtype=np.int16)).writeto(path)
            with self.assertRaises(ValueError) as raised:
                footprint_from_mask(path, (32, 32))

        assert "(32, 32)" in str(raised.exception)


class TestMaskedFractionGuard(unittest.TestCase):
    def test_an_over_cap_sightline_keeps_only_its_strongest(self):
        resid = np.zeros((1, 1, 10))
        resid[0, 0] = np.arange(10)  # channel 9 is the strongest
        mask = np.ones((1, 1, 10), dtype=bool)

        capped = cap_masked_fraction(mask, resid, 0.3)

        assert capped.sum() == 3
        assert capped[0, 0, 9] and capped[0, 0, 8] and capped[0, 0, 7]
        assert not capped[0, 0, 0]

    def test_a_sightline_under_the_cap_is_untouched(self):
        resid = np.random.default_rng(0).normal(size=(1, 1, 10))
        mask = np.zeros((1, 1, 10), dtype=bool)
        mask[0, 0, :2] = True

        np.testing.assert_array_equal(cap_masked_fraction(mask, resid, 0.5), mask)

    def test_the_cap_is_per_sightline_not_global(self):
        resid = np.tile(np.arange(10.0), (2, 1, 1))
        mask = np.zeros((2, 1, 10), dtype=bool)
        mask[0, 0] = True  # over cap
        mask[1, 0, :2] = True  # under cap

        capped = cap_masked_fraction(mask, resid, 0.3)

        assert capped[0, 0].sum() == 3
        assert capped[1, 0].sum() == 2, "an under-cap sightline must not be trimmed by its neighbour"


class TestAutomaskUnion(unittest.TestCase):
    """The union is what makes the footprint useful; check it is a union."""

    def setUp(self):
        from mowjsub.fitfuncs import FitBSpline

        self.nchan = 128
        self.freqs = 1400.0 + 0.1 * np.arange(self.nchan)
        chan = np.arange(self.nchan)
        rng = np.random.default_rng(3)

        x = (chan - self.nchan / 2) / self.nchan
        cont = 2.0 + 0.4 * x + 0.7 * x**2
        # A shallow absorber at pixel (1, 1) only.
        self.cube = np.tile(cont, (3, 3, 1)) + rng.normal(0, 0.02, (3, 3, self.nchan))
        self.cube[1, 1] -= 0.09 * np.exp(-0.5 * ((chan - 64) / 2.5) ** 2)

        self.fit = FitBSpline(self.freqs, order=3, chanwidth=16, fit_tol=0, seed=7)
        self.fit.prepare()

    def _mask(self, **kw):
        from mowjsub.fitfuncs import FitBSpline

        f = FitBSpline(self.freqs, order=3, chanwidth=16, fit_tol=0, seed=7)
        f.prepare()
        # min_island is exercised by TestIslandCut. Disabled here so these test
        # the union alone: this cube is 3x3 with no beam, so a real feature
        # cannot span sightlines the way it does under a PSF, and a marginal
        # detection is legitimately one voxel.
        kw.setdefault("min_channels", 1)
        kw.setdefault("source_min_channels", 1)
        return get_automask(self.cube, f, 3.0, **kw)

    def test_the_footprint_only_ever_adds(self):
        base = self._mask()
        fp = np.zeros((3, 3), dtype=bool)
        fp[1, 1] = True
        deeper = self._mask(source_footprint=fp, source_sigma_clip=2.0)

        assert np.all(deeper | base == deeper), "the union must never unmask what the blind clip caught"
        assert deeper.sum() > base.sum(), "a lower threshold on the absorber should catch more"

    def test_nothing_changes_outside_the_footprint(self):
        fp = np.zeros((3, 3), dtype=bool)
        fp[1, 1] = True
        base = self._mask()
        deeper = self._mask(source_footprint=fp, source_sigma_clip=2.0)

        off = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
        for i, j in off:
            np.testing.assert_array_equal(deeper[i, j], base[i, j], f"pixel {(i, j)} is outside the footprint")

    def test_an_equal_threshold_is_inert(self):
        fp = np.ones((3, 3), dtype=bool)
        np.testing.assert_array_equal(self._mask(source_footprint=fp, source_sigma_clip=3.0), self._mask())

    def test_a_footprint_of_the_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            self._mask(source_footprint=np.ones((5, 5), dtype=bool), source_sigma_clip=2.0)

        assert "(3, 3)" in str(raised.exception)

    def test_the_guard_bounds_what_a_deep_threshold_can_mask(self):
        fp = np.ones((3, 3), dtype=bool)
        capped = self._mask(source_footprint=fp, source_sigma_clip=0.5, max_masked_fraction=0.25)

        assert capped.sum(axis=-1).max() <= int(0.25 * self.nchan)


if __name__ == "__main__":
    unittest.main()


class TestComponentFilter(unittest.TestCase):
    """The two-axis cut that makes the growth discriminate signal from noise."""

    def _clip(self, data, n=3.0, **kw):
        from mowjsub.masking import Mask, PixSigmaClip

        return ~Mask(PixSigmaClip(n, **kw)).getMask(data)

    def test_a_spectrally_thin_detection_is_dropped(self):
        """Beam-correlated noise is extended on the sky but one channel deep."""
        data = np.random.default_rng(1).normal(0, 1, (9, 9, 201))
        data[3:7, 3:7, 100] = 8.0  # four beams wide, one channel deep

        assert self._clip(data, min_channels=1)[5, 5, 100], "detected with no cut"
        assert not self._clip(data, min_channels=2)[5, 5, 100], "and dropped once 2 channels are required"

    def test_a_spatially_small_detection_is_dropped(self):
        data = np.random.default_rng(2).normal(0, 1, (9, 9, 201))
        data[4, 4, 98:103] = 8.0  # one sightline, five channels deep

        assert self._clip(data, min_channels=2)[4, 4, 100], "passes the spectral cut"
        assert not self._clip(data, min_channels=2, sky_element=np.ones((2, 2), bool))[4, 4, 100], "but not the sky cut"

    def test_a_detection_extended_on_both_axes_survives_and_is_grown(self):
        data = np.random.default_rng(3).normal(0, 1, (9, 9, 201))
        data[3:7, 3:7, 98:103] = 8.0

        cut = self._clip(data, min_channels=2, sky_element=np.ones((2, 2), bool))

        assert cut[3:7, 3:7, 98:103].all(), "the component survives"
        assert cut[4, 4, 97] and cut[4, 4, 103], "and the growth still reaches its wings"

    def test_defaults_leave_the_mask_untouched(self):
        data = np.random.default_rng(4).normal(0, 1, (8, 8, 96))

        np.testing.assert_array_equal(self._clip(data), self._clip(data, min_channels=1, sky_element=None))

    def test_the_cut_removes_most_beam_correlated_noise(self):
        from scipy import ndimage

        rng = np.random.default_rng(5)
        noise = ndimage.gaussian_filter(rng.normal(0, 1, (48, 48, 192)), sigma=(1.7, 1.7, 0), mode="wrap")
        noise /= noise.std()

        before = self._clip(noise).mean()
        after = self._clip(noise, min_channels=2, sky_element=np.ones((2, 2), bool)).mean()

        assert before > 0.05, f"beam-correlated noise loses {before:.1%} of the band with no cut"
        assert after < before / 4, f"and {after:.1%} with one"


class TestBeamElement(unittest.TestCase):
    def _hdr(self, bmaj=4e-3, bmin=4e-3, cdelt=1e-3):
        hdr = _header(cdelt=cdelt)
        hdr["BMAJ"], hdr["BMIN"] = bmaj, bmin
        return hdr

    def test_a_round_beam_gives_a_round_element(self):
        from mowjsub.sources import beam_element

        element = beam_element(self._hdr(), 1.0)
        # A 4x4 pixel beam: the element is about pi * 2 * 2 = 12.6 pixels.
        assert 9 <= element.sum() <= 16, element.sum()
        assert element.shape[0] == element.shape[1]

    def test_a_smaller_fraction_gives_a_smaller_element(self):
        from mowjsub.sources import beam_element

        assert beam_element(self._hdr(), 0.5).sum() < beam_element(self._hdr(), 1.0).sum()

    def test_a_tiny_fraction_still_selects_something(self):
        from mowjsub.sources import beam_element

        assert beam_element(self._hdr(), 0.01).sum() >= 1, "rounding up must never reach an empty element"

    def test_zero_disables_it(self):
        from mowjsub.sources import beam_element

        assert beam_element(self._hdr(), 0.0) is None

    def test_an_elongated_beam_is_oriented_by_bpa(self):
        from mowjsub.sources import beam_element

        hdr = self._hdr(bmaj=8e-3, bmin=2e-3)
        hdr["BPA"] = 0.0  # major axis along +Dec, i.e. the y/dec pixel axis
        along_dec = beam_element(hdr, 1.0)
        hdr["BPA"] = 90.0  # major axis along East, i.e. the x/ra pixel axis
        along_ra = beam_element(hdr, 1.0)

        # Same area either way, but the long axis swaps between the two.
        assert abs(int(along_dec.sum()) - int(along_ra.sum())) <= 2
        dec_extent = along_dec.any(axis=0).sum(), along_dec.any(axis=1).sum()
        ra_extent = along_ra.any(axis=0).sum(), along_ra.any(axis=1).sum()
        assert dec_extent[0] > dec_extent[1], f"BPA 0 should be long in dec, got {dec_extent}"
        assert ra_extent[1] > ra_extent[0], f"BPA 90 should be long in ra, got {ra_extent}"

    def test_a_header_with_no_beam_says_so(self):
        from mowjsub.sources import beam_element

        with self.assertRaises(RuntimeError) as raised:
            beam_element(_header(), 0.5)

        assert "BMAJ" in str(raised.exception)


if __name__ == "__main__":
    unittest.main()
