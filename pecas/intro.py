#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def pecas_intro():

    try:

        os.environ["PECAS_INTRO_SHOWN"]

    except:

        os.environ["PECAS_INTRO_SHOWN"] = "1"

        # print('\n' + 78 * '-')
        print('\n' + 30 * '-' + ' Welcome at PECas ' + 30 * '-')
        print('\n' + 35 * ' ' + ' PECas ' + 36 * ' ')
        print(21 * ' ' + ' Parameter estimation using CasADi ' + 22 * ' ')
        print(14 * ' ' + \
            ' Adrian Buerger and Jesus Lago Garcia, 2014-2015 ' + 13 * ' ')
        print(19 * ' ' + \
            ' SYSCOP, IMTEK, University of Freiburg ' + 20 * ' ')
