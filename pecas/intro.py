#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

import os

def pecas_intro():

    try:

        os.environ["PECAS_INTRO_SHOWN"]

    except:

        os.environ["PECAS_INTRO_SHOWN"] = "1"

        # print('\n' + 78 * '-')
        print('\n' + 27 * '-' + ' Welcome to PECas 0.5.1' + 28 * '-')
        print('\n' + 35 * ' ' + ' PECas ' + 36 * ' ')
        print(21 * ' ' + ' Parameter estimation using CasADi ' + 22 * ' ')
        print(14 * ' ' + \
            ' Adrian Buerger and Jesus Lago Garcia, 2014-2015 ' + 13 * ' ')
        print(19 * ' ' + \
            ' SYSCOP, IMTEK, University of Freiburg ' + 20 * ' ')
