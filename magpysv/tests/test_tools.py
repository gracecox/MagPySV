# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:45:42 2017

Testing functions for tools.py.

@author: Grace Cox and Will Brown
"""
import unittest
import os
from .. import tools
import pandas as pd
import datetime as dt


class DataResamplingTestCase(unittest.TestCase):
    """Set up test case for data resampling"""
    def setUp(self):
        """Specify location of test file"""
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        testfile = os.path.join(self.path, 'testdaily.csv')
        self.col_names = ['date', 'code', 'component', 'daily_mean']
        self.data = pd.read_csv(testfile, sep=' ', header=0,
                                names=self.col_names, parse_dates=[0])
        self.averaged = tools.data_resampling(self.data)

    def test_data_resampling(self):
        """Verify correct resampling of test file data"""
        self.assertAlmostEqual(self.averaged.daily_mean.values[0], 801.000000)
        self.assertAlmostEqual(self.averaged.daily_mean.values[7],
                               33335.750000)
        self.assertAlmostEqual(self.averaged.daily_mean.values[-1],
                               45115.500000)
        self.assertEqual(self.averaged.date[0], dt.datetime(day=15, month=1,
                         year=2000))
        self.assertEqual(self.averaged.date[1], dt.datetime(day=15, month=2,
                         year=2000))
        self.assertEqual(self.averaged.date[7], dt.datetime(day=15, month=8,
                         year=2000))
