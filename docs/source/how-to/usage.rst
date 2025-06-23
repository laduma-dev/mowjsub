.. _usage:

.. role:: raw-math(raw)
    :format: latex html

Command-line Applications
=================================

:command:`imcontsub`
---------------------
The image-plane application :command:`imcontsub` is used to perform continuum subtraction on spectral line data cubes. It can handle large data cubes efficiently by processing them in chunks. The chunking is done along the RA-axis, i.e, the data cube is split into smaller sub-cubes that are processed in parallel.


A key advantage of image-plane continuum subtraction is that it allows for robust on-the-fly thresholding due to the high signal-to-noise ratio compared to the uv-plane. This is particularly useful on for data cubes unknown line emission, which, if not accounted for, can lead to significant errors in the continuum subtraction process. This on-the-fly thresholding can be enabled by setting the ``--sigma-clip`` parameter.

.. code-block:: bash

    imcontsub --output-prefix output_prefix \
                --sigma-clip 5 \
                --order 3 \
                --segments 250 \
                --ra-chunks 64 \
                --nworkers 8 \
                input_fits_cube.fits

However, if the line emission is known, a binary mask can be provided to the ``--mask-image`` option. In this case, the ``--sigma-clip`` parameter will be ignored (if set.)

:command:`imcontsub` also allows the user to specify multiple ``--sigma-clip``, ``--order`` and ``--segments`` parameters. This allows the user to do the continuum subtraction in multiple iterations, each time using a different set of parameters. The advantage of this is that the user can start with a large ``--sigma-clip`` and wide ``--segments`` values to remove the most significant line emission, and then gradually decrease both to remove smaller line emission features. When using this mode, the ``--segments`` and ``--order`` must have the same length. Here's an example

.. code-block:: bash

    imcontsub --output-prefix output_prefix \
                --sigma-clip 5 5 3 \
                --order 3 2 2 \
                --segments 400 300 250 \
                --ra-chunks 64 \
                --nworkers 8 \
                input_fits_cube.fits


