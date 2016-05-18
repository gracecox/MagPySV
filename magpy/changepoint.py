# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program.  If not, see <http://www.gnu.org/licenses/>."""
"""
Created on Tue Feb 16 17:22:26 2016

@author: Grace
"""

import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
cpm = importr('cpm')


def change_point_analysis(signal, cpm_method):
    res = cpm.detectChangePoint(
        ro.FloatVector(signal), cpmType=cpm_method,
        ARL0=500, startup=20)
    # Convert the ListVector returned by cpm to a python dictionary
    results = {key: res.rx2(key)[0] for key in res.names}

    return results


def plot_cpa_results(dates, signal, results):
    if results['changeDetected']:
        plt.figure(figsize=(10, 7))
        plt.gca().xaxis_date()
        plt.plot(dates, signal, 'b')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('SV (nT/yr)', fontsize=16)
        # Draw vertical lines at each change point location
        if isinstance(results['changePoint'], int):     # Only 1 change point
            plt.axvline(dates[results['changePoint']], color='r',
                        linestyle='--')
        else:
            for chgpoint in results['changePoint']:  # More than 1 change point
                plt.axvline(dates[chgpoint], color='r', linestyle='--')
