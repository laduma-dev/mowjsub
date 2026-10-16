.. _installation:


Installation
=============
The stable version Mowj-sub is available on `PyPI <https://pypi.org/project/mowjsub>`_ and can be installed using:

.. code-block:: bash

    pip install mowjsub

The latest (in development) version can be installed directly from GitHub by using:

.. code-block:: bash

    pip install git+https://github.com/laduma-dev/mowjsub.git

Optional Stimela integration
---------------------------

The CLI and library use the base installation. Install the optional extra to
use the steps from Stimela 3 / Ninja::

    pip install 'mowjsub[ninja]'

For development, use ``uv sync --extra ninja`` to include integration tests.

The base installation requires fitstoolz 0.2.2 or newer, which also makes
Stimela optional. Both local development and wheel installations use the
published dependency.
