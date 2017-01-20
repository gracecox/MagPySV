# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:26:22 2016

@author: Grace
"""
import unittest
import os
from .. import inputoutput
from .. import svtools
import pandas as pd
import datetime as dt
import numpy as np


class WDCParsefileTestCase(unittest.TestCase):

    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_wdc_parsefile1(self):

        testfile = os.path.join(self.path, 'testdata1.wdc')

        data = inputoutput.wdc_parsefile(testfile)
        # Observatory code
        self.assertEqual(data.code[0], 'NGK')
        self.assertEqual(len(data.code.unique()), 1)
        # Components
        self.assertTrue(any(
            x in {'X', 'Y', 'Z', 'D', 'I', 'H'} for x in data.component))
        self.assertEqual(data.component[3], 'D')
        self.assertEqual(data.component.values[-1], 'Z')

    def test_wdc_parsefile2(self):

        testfile = os.path.join(self.path, 'testdata2.wdc')

        data = inputoutput.wdc_parsefile(testfile)
        # Observatory code
        self.assertEqual(data.code[0], 'NGK')
        self.assertEqual(len(data.code.unique()), 1)
        # Components
        self.assertTrue(any(
            x in {'X', 'Y', 'Z', 'D', 'I', 'H'} for x in data.component))
        self.assertEqual(data.component[3], 'D')
        self.assertEqual(data.component.values[-1], 'Z')


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

        df = inputoutput.wdc_datetimes(self.data)

        self.assertTrue(isinstance(df.date[0], pd.datetime))
        self.assertEqual(df.date[0], dt.datetime(day=15, month=1, year=1963))


class DailyMeanConversionTestCase(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame(
            index=[0, 1], columns=[
                'date', 'component', 'base', 'daily_mean_temp'])
        self.data.component = ['I', 'X']
        self.data.base = [53, 200]
        self.data.daily_mean_temp = [1200, 530]

    def test_daily_mean_conversion(self):

        df = inputoutput.daily_mean_conversion(self.data)

        self.assertAlmostEqual(df.iloc[0].daily_mean, 55)
        self.assertAlmostEqual(df.iloc[1].daily_mean, 20530)


class AnglesToGeographicTestCase(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame(
            index=[0, 1], columns=[
                'date', 'component', 'daily_mean'])
        self.data.component = ['H', 'D']
        self.data.daily_mean = [20530, 55]
        self.data.date = [dt.date(day=15, month=1, year=1963), dt.date(day=15,
                          month=1, year=1963)]
        self.data = self.data.pivot(index='date', columns='component',
                                    values='daily_mean')

    def test_angles_to_geographic(self):

        df = inputoutput.angles_to_geographic(self.data)

        self.assertAlmostEqual(df.iloc[0].X, 11775.524238286978)
        self.assertAlmostEqual(df.iloc[0].Y, 16817.191469253001)


class WDCXYZTestCase(unittest.TestCase):

    def setUp(self):

        self.data = pd.DataFrame(
            index=[0, 1, 2, 3, 4, 5], columns=[
                'date', 'component', 'base', 'daily_mean_temp'])
        self.data.component = ['H', 'D', 'X', 'Y', 'Z', 'X']
        self.data.base = [200, 53, np.nan, np.nan, 300, 9999]
        self.data.daily_mean_temp = [530, 1200, np.nan, np.nan, 430, 9999]
        self.data.date.iloc[0:5] = dt.date(day=15, month=1, year=1963)
        self.data.date.iloc[5] = dt.date(day=20, month=1, year=1963)

    def test_wdc_xyz(self):

        df = inputoutput.wdc_xyz(self.data)

        self.assertAlmostEqual(df.iloc[0].X, 11775.524238286978)
        self.assertAlmostEqual(df.iloc[0].Y, 16817.191469253001)
        self.assertAlmostEqual(df.iloc[0].Z, 30430.000000000000)
        self.assertTrue(np.isnan(df.iloc[1].X))


class DataResamplingTestCase(unittest.TestCase):

    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        testfile = os.path.join(self.path, 'testdaily.csv')
        self.col_names = ['date', 'code', 'component', 'daily_mean']
        self.data = pd.read_csv(testfile, sep=' ', header=0,
                                names=self.col_names, parse_dates=[0])
        self.averaged = svtools.data_resampling(self.data)

    def test_data_resampling(self):

        self.assertAlmostEqual(self.averaged.daily_mean.values[0], 801.000000)
        self.assertAlmostEqual(self.averaged.daily_mean.values[7],
                               33335.750000)
        self.assertAlmostEqual(self.averaged.daily_mean.values[-1],
                               45115.500000)
        self.assertEqual(self.averaged.date[0], dt.datetime(day=1, month=1,
                         year=2000))
        self.assertEqual(self.averaged.date[1], dt.datetime(day=1, month=2,
                         year=2000))
        self.assertEqual(self.averaged.date[7], dt.datetime(day=1, month=8,
                         year=2000))
