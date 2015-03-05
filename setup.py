#!/usr/bin/env python
# -*- coding: utf-8 -*-

# todo: license & copywrite

from setuptools import setup

# todo: license, update email, check dependencies

setup(
    name='pecas',
    version='0.0.1',
    author='Adrian Buerger',
    author_email='adrian.buerger@hs-karlsruhe.de',
    packages=['pecas'],
    package_dir={'pecas': 'pecas'},
        url='http://github.com/adbuerger/PECas/',
    license='LGPL',
    zip_safe=False,
    description='Parameter estimation using CasADi in Python',
    install_requires=[],
    use_2to3=True,
)
