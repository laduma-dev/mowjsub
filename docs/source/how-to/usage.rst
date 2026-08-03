.. _usage:

.. role:: raw-math(raw)
    :format: latex html

Command-line Applications
=================================

:command:`im-mowjsub`
---------------------
The image-plane application :command:`im-mowjsub` is used to perform continuum subtraction on spectral line data cubes. It can handle large data cubes efficiently by processing them in chunks. The chunking is done along the RA-axis, i.e, the data cube is split into smaller sub-cubes that are processed in parallel.


A key advantage of image-plane continuum subtraction is that it allows for robust on-the-fly thresholding due to the high signal-to-noise ratio compared to the uv-plane. This is particularly useful on for data cubes unknown line emission, which, if not accounted for, can lead to significant errors in the continuum subtraction process. This on-the-fly thresholding can be enabled by setting the ``--sigma-clip`` parameter.

.. code-block:: bash

    im-mowjsub --output-prefix output_prefix \
                --sigma-clip 5 \
                --order 3 \
                --segments 250 \
                --ra-chunks 64 \
                --nworkers 8 \
                input_fits_cube.fits

However, if the line emission is known, a binary mask can be provided to the ``--mask-image`` option. In this case, the ``--sigma-clip`` parameter will be ignored (if set.)

:command:`im-mowjsub` also allows the user to specify multiple ``--sigma-clip``, ``--order`` and ``--segments`` parameters. This allows the user to do the continuum subtraction in multiple iterations, each time using a different set of parameters. The advantage of this is that the user can start with a large ``--sigma-clip`` and wide ``--segments`` values to remove the most significant line emission, and then gradually decrease both to remove smaller line emission features. When using this mode, the ``--segments`` and ``--order`` must have the same length. Here's an example

.. code-block:: bash

    im-mowjsub --output-prefix output_prefix \
                --sigma-clip 5 5 3 \
                --order 3 2 2 \
                --segments 400 300 250 \
                --ra-chunks 64 \
                --nworkers 8 \
                input_fits_cube.fits


:command:`vis-mowjsub`
----------------------
The visibility-plane application :command:`vis-mowjsub` performs continuum subtraction directly on a Measurement Set. It fits a continuum baseline to every spectrum, per baseline and per correlation, and writes the residual to ``--output-column`` of the input MS, leaving the input data untouched.

Passing ``--output-ms`` instead leaves the input MS entirely alone and writes a complete new MS, with all subtables copied across and the line data in ``--output-column``. The visibilities the fit was made against are not duplicated into it, so the result is roughly the size of one data column.

.. code-block:: bash

    vis-mowjsub input.ms \
                --fit-model b-spline \
                --order 3 \
                --vel-width 250 \
                --output-column LINE_DATA \
                --nworkers 8


Doppler correction
^^^^^^^^^^^^^^^^^^
The channel frequencies recorded in an MS are topocentric: they are fixed relative to the telescope, so the sky frequency each channel corresponds to drifts as the Earth turns and moves along its orbit. Spectral line work needs a channel grid fixed relative to the source instead, which is what CASA ``mstransform`` produces with ``regridms=True``.

Setting ``--doppler-frame`` makes :command:`vis-mowjsub` do that resampling itself, removing the need for a separate ``mstransform`` pass over the data:

.. code-block:: bash

    vis-mowjsub input.ms \
                --fit-model b-spline \
                --order 3 \
                --vel-width 250 \
                --output-column DATA \
                --doppler-frame bary \
                --output-ms line.ms

The continuum is always fitted **before** the regrid, on the native topocentric grid. This is deliberate: the bandpass and standing-wave structure the fit is modelling is stationary in topocentric frequency, and fitting first also means the fit sees channels whose noise is still uncorrelated. Only the residual is moved onto the Doppler-corrected grid. Note that this ordering differs from CASA ``mstransform``, which regrids before it subtracts.

Because the output channel grid differs from the input one, ``--doppler-frame`` requires ``--output-ms``; the result cannot be written back into a column of the input MS. The new MS carries the regridded line data in ``--output-column``, together with a ``SPECTRAL_WINDOW`` describing the new grid and its reference frame.

``--output-column`` is required and has no default: there is no standard MS column name for continuum-subtracted visibilities, so mowjsub does not invent one. ``DATA`` is usually the right answer when the result goes straight to an imager. Note that without ``--output-ms`` the column is written back into the input MS, so naming the column you are reading is refused rather than allowed to destroy it.

Available frames are ``topo``, ``geo``, ``bary``, ``lsrk``, ``lsrd``, ``galacto``, ``lgroup``, ``cmb`` and ``source``, matching the options CASA accepts. ``bary`` and ``lsrk`` are the usual choices for extragalactic and Galactic HI respectively. The frame velocities and the way conversion steps are composed follow casacore, so the grids agree with CASA's to well under 0.1 m/s. ``source`` additionally needs a systemic velocity, taken from the MS ``SOURCE::SYSVEL`` column or from ``--doppler-source-vel`` in km/s.

By default (``--doppler-chan-grid auto``) the output grid is derived from the observation: the range of sky frequencies covered at *every* timestamp, with one guard channel dropped at each end so no output channel ever falls outside the observed band. Since :command:`vis-mowjsub` processes one MS at a time, ``auto`` cannot align several MSs with each other. To put a set of observations on one common grid, compute it once and pass it explicitly:

.. code-block:: bash

    vis-mowjsub input.ms \
                --output-column DATA \
                --doppler-frame bary \
                --doppler-chan-grid '1000,1419.5MHz,26.1kHz' \
                --output-ms line.ms

The grid is given as ``nchan,chan0,chanwidth``; both frequencies need a unit (``Hz``, ``kHz``, ``MHz`` or ``GHz``), and ``chanwidth`` is negative for a descending band.

Resampling uses nearest-neighbour by default, which is what caracal asks of ``mstransform`` and which leaves the noise in neighbouring channels uncorrelated. ``--doppler-interpolation linear`` is smoother but correlates adjacent channels. In either case an output channel is flagged if it falls outside the observed band at that timestamp or if any input channel feeding it is flagged, and the output weights follow from propagating the input variances through the interpolation.



Doppler correction as a separate step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:command:`vis-mowjsub --doppler-frame` fuses the two operations into one pass, which is convenient but forces a pipeline to reach the correction *through* a continuum subtraction. :command:`doppler-mowjsub` exposes the same correction on its own, over an MS whose continuum has already gone:

.. code-block:: bash

    doppler-mowjsub line.ms \
                --input-column LINE_DATA \
                --output-column DATA \
                --doppler-frame bary \
                --output-ms out.ms

It takes the same ``--doppler-*`` parameters, does no fitting, and produces exactly what the fused path produces — the two are pinned against each other in the test suite. Use it when continuum subtraction and the frame transformation need to be separate pipeline stages, for instance where a workflow would otherwise run CASA ``mstransform`` after the subtraction and get the order the wrong way round.

``--input-column`` and ``--output-column`` are both required. There is no standard MS column for continuum-subtracted visibilities, so the input column is whatever the subtraction stage was told to write, and guessing it is how you Doppler-correct the wrong data in silence.

The input must still be on its native topocentric channel grid; an MS that has already been regridded, by ``mstransform`` or by an earlier mowjsub run, is refused rather than corrected twice. mowjsub reads ``SPECTRAL_WINDOW::MEAS_FREQ_REF`` to decide.

Doppler correction in the image plane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:command:`im-mowjsub` also accepts ``--doppler-frame``, applied to both output cubes after the continuum fit:

.. code-block:: bash

    im-mowjsub cube.fits \
               --output-prefix out \
               --fit-model b-spline --order 3 --vel-width 250 \
               --doppler-frame bary \
               --doppler-obs-duration 2.5

**Read this before using it.** The Doppler factor changes as the Earth turns, so the visibility-plane commands apply one factor per timestamp. A cube has already been integrated over time, so only a single factor can be applied to it. Shifting the spectral axis re-centres the line but cannot undo the smearing the drift caused while the data were being imaged — that information was destroyed by the integration.

The test is a ratio:

    the image-plane correction is safe when the line-of-sight velocity drifts by much less than one channel over the observation.

Both quantities scale with frequency, so the ratio is the same whether you state it in km/s or kHz. Pass ``--doppler-obs-duration`` (in hours) and mowjsub computes the drift, logs it against the channel width, and warns when it exceeds a tenth of a channel. Without that option the correction is still applied, at the epoch in the header, but the check cannot be made and mowjsub says so. For a long track or a narrow channelisation, correct in the visibility plane instead.

Where the image-plane path is unambiguously right is putting several observations onto one grid for stacking. That is a pure shift per cube, with no smearing penalty at all — give each run the same ``--doppler-chan-grid``.

A cube carries far less metadata than an MS, so the correction is resolved from the header with explicit overrides for what it cannot supply:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Taken from
     - Override
   * - direction
     - the celestial WCS reference coordinate
     - ``--doppler-phase-centre 'RA,Dec'``
   * - epoch
     - ``MJD-OBS``, else ``DATE-OBS``
     - ``--doppler-time`` (ISO or MJD)
   * - track length
     - nothing standard records it
     - ``--doppler-obs-duration``
   * - position
     - ``OBSGEO-X/Y/Z``, else ``TELESCOP``
     - ``--doppler-telescope``

Both the continuum and line cubes are written on the corrected grid, so the pair stays recombinable, and the spectral WCS is rewritten with the new ``SPECSYS``. Stale velocity keywords (``ALTRVAL``, ``ALTRPIX``, ``VELREF``) are dropped rather than left describing the old grid. ``--doppler-frame source`` needs ``--doppler-source-vel``, since a cube has no ``SOURCE::SYSVEL`` to fall back on.

The spectral axis must be ``NAXIS3``, i.e. the axis order must be ``RA, DEC, FREQ[, STOKES]``. A cube with STOKES on ``NAXIS3`` and FREQ on ``NAXIS4`` is refused with a message saying so, whether or not a Doppler correction was asked for -- that layout is unsupported throughout :command:`im-mowjsub`, not just here. Reorder the axes (CASA ``imtrans``) and try again.
