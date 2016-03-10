# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:26:22 2016

@author: Grace
"""
import unittest
import os
import wdc_io
import pandas as pd


class WDCReadfileTestCase(unittest.TestCase):

    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_wdc_readfile(self):

        testfile = os.path.join(self.path, 'testdata.wdc')

        data = wdc_io.wdc_readfile(testfile)
        # Observatory code
        self.assertEqual(data.code[0], 'NGK')
        self.assertEqual(len(data.code.unique()), 1)
        # Components
        self.assertTrue(any(
            x in {'X', 'Y', 'Z', 'D', 'I', 'H'} for x in data.component))
        self.assertEqual(data.component[3], 'D')
        self.assertEqual(data.component.values[-1], 'Z')
#        # Daily mean (using the tabular base)
#        self.assertAlmostEqual(data.daily_mean.values[-1],45113)
#        self.assertAlmostEqual(data.daily_mean[3],800)


class WDCDatetimesTestCase(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame(
            index=[0], columns=['century', 'yr', 'month', 'day', 'code'])
        self.data['century'] = 19
        self.data['yr'] = 63
        self.data['month'] = 1
        self.data['day'] = 15
        self.data['code'] = 'NGK'

    def test_wdc_datetimes(self):

        df = wdc_io.wdc_datetimes(self.data)

        self.assertTrue(isinstance(df.date[0], pd.datetime))


# =============================================================================
# class DatatAveragingTestCase(unittest.TestCase):
#     def setUp(self):
#         # Directory where the test files are located
#         self.path = os.path.join(os.path.dirname(__file__), 'data')
#         testfile = os.path.join(self.path, 'testdata.wdc')
#         self.rawdata = wdc_io.wdc_readfile(testfile)
#         self.data = wdc_io.wdc_datetimes(self.rawdata)
#
#     def test_wdc_data_averaging(self):
#
#         self.averaged = self.data.apply(wdc_io.data_averaging)
#         # Daily mean (using the tabular base)
#         self.assertAlmostEqual(self.averaged.daily_mean.values[-1], 45113)
#         self.assertAlmostEqual(self.averaged.daily_mean[3], 800)
#
#
# class DailyToMonthlyTestCase(unittest.TestCase):
#
#     def setUp(self):
#         # Directory where the test files are located
#         self.path = os.path.join(os.path.dirname(__file__), 'data')
#
#     def test_wdc_to_dataframe(self):
#
#         testfile = os.path.join(self.path, 'testdaily.csv')
#
#         data = wdc_io.wdc_readfile(testfile)
#==============================================================================
