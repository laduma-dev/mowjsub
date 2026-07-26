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
                --doppler-frame bary \
                --output-ms line.ms

The continuum is always fitted **before** the regrid, on the native topocentric grid. This is deliberate: the bandpass and standing-wave structure the fit is modelling is stationary in topocentric frequency, and fitting first also means the fit sees channels whose noise is still uncorrelated. Only the residual is moved onto the Doppler-corrected grid. Note that this ordering differs from CASA ``mstransform``, which regrids before it subtracts.

Because the output channel grid differs from the input one, ``--doppler-frame`` requires ``--output-ms``; the result cannot be written back into a column of the input MS. The new MS carries the regridded line data in ``--output-column``, together with a ``SPECTRAL_WINDOW`` describing the new grid and its reference frame. Pass ``--output-column DATA`` if the imager you feed it to expects that column.

Available frames are ``topo``, ``geo``, ``bary``, ``lsrk``, ``lsrd``, ``galacto``, ``lgroup``, ``cmb`` and ``source``, matching the options CASA accepts. ``bary`` and ``lsrk`` are the usual choices for extragalactic and Galactic HI respectively. The frame velocities and the way conversion steps are composed follow casacore, so the grids agree with CASA's to well under 0.1 m/s. ``source`` additionally needs a systemic velocity, taken from the MS ``SOURCE::SYSVEL`` column or from ``--doppler-source-vel`` in km/s.

By default (``--doppler-chan-grid auto``) the output grid is derived from the observation: the range of sky frequencies covered at *every* timestamp, with one guard channel dropped at each end so no output channel ever falls outside the observed band. Since :command:`vis-mowjsub` processes one MS at a time, ``auto`` cannot align several MSs with each other. To put a set of observations on one common grid, compute it once and pass it explicitly:

.. code-block:: bash

    vis-mowjsub input.ms \
                --doppler-frame bary \
                --doppler-chan-grid '1000,1419.5MHz,26.1kHz' \
                --output-ms line.ms

The grid is given as ``nchan,chan0,chanwidth``; both frequencies need a unit (``Hz``, ``kHz``, ``MHz`` or ``GHz``), and ``chanwidth`` is negative for a descending band.

Resampling uses nearest-neighbour by default, which is what caracal asks of ``mstransform`` and which leaves the noise in neighbouring channels uncorrelated. ``--doppler-interpolation linear`` is smoother but correlates adjacent channels. In either case an output channel is flagged if it falls outside the observed band at that timestamp or if any input channel feeding it is flagged, and the output weights follow from propagating the input variances through the interpolation.


