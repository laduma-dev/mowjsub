Mowjsub
=======

.. image:: https://img.shields.io/pypi/v/mowjsub.svg
    :target: https://pypi.org/project/mowjsub/
    :alt: PyPI version

.. image:: https://readthedocs.org/projects/mowjsub/badge/?version=latest
    :target: https://mowjsub.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


Image and uv-plane continuum subtraction tools for spectral line data.

Installation
------------

PyPI (stable; recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install mowjsub

Latest (in development)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install git+https://github.com/laduma-dev/mowjsub.git


Stimela 3 integration is optional. The three console commands and Python
library use the base install. To use mowjsub steps in Ninja recipes::

    pip install 'mowjsub[ninja]'

The base installation requires fitstoolz 0.2.2 or newer, which also makes
Stimela optional. Both local development and wheel installations use the
published dependency.

Existing targets such as ``mowjsub.parser.im_mowjsub:step`` and
``mowjsub.parser.im_mowjsub:im_mowjsub`` resolve to genuine Stimela StepRefs
when the extra is installed. The visibility and Doppler modules expose the
same two forms.

Using uv (Recommended development install)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The package requires Python >=3.11, so uv is the recommended way to install
from a local clone. If you do not have `uv <https://docs.astral.sh/uv/>`_ installed, install it first:

.. code-block:: bash

    pip install uv

Then clone the repository, navigate into it, activate your existing virtual
environment, and sync dependencies:

.. code-block:: bash

    git clone https://github.com/laduma-dev/mowjsub.git
    cd mowjsub

    source /path/to/your/.venv/bin/activate

    uv sync --python 3.12 --active 
.. note::

    ``--active`` ensures the package is installed in your activated virtual environment.

The full documentation is available on `readthedocs <https://mowjsub.readthedocs.io/>`_.


