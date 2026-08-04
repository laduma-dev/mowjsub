.. _reference:

Reference
###########
.. role:: raw-math(raw)
    :format: latex html


.. Point sphinx-click at ``command``, not ``runit``. Since the port to Stimela 3
.. ``runit`` is a plain function taking an options namespace, and the
.. ``click.Command`` the console script installs is built from the pystep
.. separately. Naming ``runit`` here is not a broken link but an empty page:
.. sphinx-click raises "is not click.Command or click.Group" and drops the whole
.. directive, and the docs still build.

.. click:: mowjsub.parser.im_mowjsub:command
    :prog: im-mowjsub
    :nested: full


.. click:: mowjsub.parser.vis_mowjsub:command
    :prog: vis-mowjsub
    :nested: full


.. click:: mowjsub.parser.doppler_mowjsub:command
    :prog: doppler-mowjsub
    :nested: full


