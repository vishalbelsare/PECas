Installing PECas
================

In order to use PECas, please make sure that
`Python <https://www.python.org/>`_ as well as
`Python Numpy <http://www.numpy.org/>`_
and a recent version of `CasADi <https://github.com/casadi/casadi/wiki>`_ are
installed on your system. PECas has only been 
run on Ubuntu Linux systems so far, and has not been tested for Windows.

If you meet these requirements, you can run PECas directly from within the 
directory in which you obtaind it, or copy ``pecas.py`` to another destination.
If you
want to use PECas from multiple locations and not only from the containing
directory, consider adding the PECas directory to your PYTHONPATH by running

.. code:: bash
    
    export PYTHONPATH=$PYTHONPATH:/path/to/PECas

inside your shell. If you also want to permanently add PECas to your
PYTHONPATH, you can e. g. on Ubuntu add the above line at the end of your
``.bashrc`` file, which is usually located in your home directory.
