Get and install PECas
=====================

Within the next sections, you will get to know how to obtain and install PECas,
and what prerequisites have to be met in order to get PECas to work correctly.

Linux installation
------------------

Prerequesites
~~~~~~~~~~~~~

In order to use PECas, please make sure that
`Python <https://www.python.org/>`_ (currently supported version is Python 2.7) as well as
`Python Numpy <http://www.numpy.org/>`_ (>= 1.8) and
`PyLab <http://wiki.scipy.org/PyLab>`_ are installed on your system. On Ubuntu 14.04, this can easily be ensured by running

.. code:: bash

    sudo apt-get update
    sudo apt-get install python python-numpy python-scipy python-matplotlib

Note that you need root privileges to do this. Also, a recent version of `CasADi <http://casadi.org>`_ (>= 2.4.0-rc2) needs to be installed on your system. You can obtain a recent version from the `CasADi web page <http://casadi.org>`_, and follow the installation instructions there.

Get PECas
~~~~~~~~~

The preferred way to obtain PECas is `directly from its
git repository <https://github.com/adbuerger/PECas>`_. You can then either clone the repository, or download the current files within a zip-archive. To obtain the zip-file you do not need to have `git <http://git-scm.com/>`_ installed, but cloning the repository provides an easy way to receive updates on PECas by pulling from the repository in regular intervals.

On Ubuntu 14.04 again, you can install git and obatin PECas using the following commands

.. code:: bash

    sudo apt-get update
    sudo apt-get install git
    git clone git@github.com:adbuerger/PECas.git

Install PECas
~~~~~~~~~~~~~

You can install PECas on Ubuntu by running the command

.. code:: bash
    
    sudo python setup.py install

from within the PECas directory. Note that you need root privileges to be able to do this. If you are planning to install PECas on systems different from Ubuntu, this command might need to be adapted.

Update PECas
~~~~~~~~~~~~

If you recieved PECas by cloning the git repository, you can update the contents of your local copy by running

.. code:: bash
    
    git pull

from within the PECas directory. Afterwards, you need to install the recent version by running

.. code:: bash
    
    sudo python setup.py install

again. Note that due to the fact that PECas is still in a very early state, an update might lead to changes e. g. in syntax and functionalities, causing that some code that was written for older versions of PECas might not work anymore with future version, and will then need to be adjusted accordingly.


Windows installation
--------------------

Prerequesites
~~~~~~~~~~~~~

The easiest way to meet the prerequesites to use PECas and CasADi on a Windows system might be to install a recent version of `Python(x,y) <http://python-xy.github.io/>`_, which is also recommended by the CasADi developers. After installation of Python(x,y), obtain a recent version of CasADi (>= 2.4.0-rc2) from the `CasADi web page <http://casadi.org>`_, and follow the installation instructions there.

Get and install PECas
~~~~~~~~~~~~~~~~~~~~~

More expereinced users can follow the guide from the Linux section to install PECas. For less experienced users, it is recommended to visit the `PECas GitHub page <https://github.com/adbuerger/PECas>`_ and download PECas as a zip archive. Just unpack the archive, start your Python interpreter or develpment environment (which with Python(x,y) would preferrably be Spyder), and add the contained PECas folder to your path, which works similar to the installation of CasADi with:

.. code:: python

    >>> import sys
    >>> sys.path.append("pecasdirectory")


Update PECas
~~~~~~~~~~~~

If you obtained PECas via git, please refer to the Linux section above on how to update PECas. If you installed PECas by adding it to your path, you can simply obtain the newest version in a zip archive again, and add the new, unpacked folder to your path as described above.

Note that due to the fact that PECas is still in a very early state, an update might lead to changes e. g. in syntax and functionalities, causing that some code that was written for older versions of PECas might not work anymore with future version, and will then need to be adjusted accordingly.


Recommendations
---------------

To speed up computations in PECas, it is recommended to install `HSL for IPOPT <http://www.hsl.rl.ac.uk/ipopt/>`_. On how to install the solvers and further information, see the page `Obtaining HSL <https://github.com/casadi/casadi/wiki/Obtaining-HSL>`_ in the CasADi wiki.