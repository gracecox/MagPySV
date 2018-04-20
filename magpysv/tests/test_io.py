# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:26:22 2016

Testing the file io functionality of io.py.

@author: Grace Cox and Will Brown
"""

import unittest
import mock
from ddt import ddt, data, unpack
from io import StringIO  # io
import os
from .. import io  # magpysv.io
from pandas.util.testing import assert_frame_equal
import pandas as pd
import datetime as dt
import numpy as np

# Directory where the test files are located
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


@ddt
class WDCParsefileTestCase(unittest.TestCase):
    """Set up for test case for parsing function (opens test datafiles)"""
    @data({'filename': 'testdata1.wdc', 'code': 'NGK', 'component1': 'D',
           'component2': 'Z'},
          {'filename': 'testdata2.wdc', 'code': 'NGK', 'component1': 'D',
          'component2': 'Z'},
          {'filename': 'testdata3.wdc', 'code': 'PSM', 'component1': 'H',
          'component2': 'D'})
    @unpack
    def test_wdc_parsefile_newformat(self, filename, code, component1,
                                     component2):
        """"Verify correct parsing of the test WDC format files"""
        testfile = os.path.join(TEST_DATA_PATH, filename)

        data = io.wdc_parsefile(testfile)
        # Observatory code
        self.assertEqual(data.code[0], code)
        self.assertEqual(len(data.code.unique()), 1)
        # Components
        self.assertTrue(any(
            x in {'X', 'Y', 'Z', 'D', 'I', 'H'} for x in data.component))
        self.assertEqual(data.component[3], component1)
        self.assertEqual(data.component.values[-1], component2)


class WDCDatetimesTestCase(unittest.TestCase):
    """Set up the test case for datetimes function"""
    def setUp(self):
        """Create a dataframe containing mock input for datetime function"""
        self.data = pd.DataFrame(
            index=[0], columns=['century', 'yr', 'month', 'day', 'hour'])
        self.data['century'] = 19
        self.data['yr'] = 88
        self.data['month'] = 9
        self.data['day'] = 21
        self.data['hour'] = 2
        self.data['code'] = 'ESK'

    def test_wdc_datetimes(self):
        """Verify correct datetime creation from mock data"""
        df = io.wdc_datetimes(self.data)

        self.assertTrue(isinstance(df.date[0], pd.datetime))
        self.assertEqual(df.date[0], dt.datetime(day=21, month=9, year=1988,
                                                 hour=2, minute=30))


class HourlyMeanConversionTestCase(unittest.TestCase):
    """Set up the test case for hourly means conversion function"""
    def setUp(self):
        """Create a dataframe containing mock input"""
        self.data = pd.DataFrame(
            index=[0, 1], columns=[
                'date', 'component', 'base', 'hourly_mean_temp'])
        self.data.component = ['I', 'X']
        self.data.base = [53, 200]
        self.data.hourly_mean_temp = [1200, 530]

    def test_hourly_mean_conversion(self):
        """Verify correct conversion to hourly means from mock data"""
        df = io.hourly_mean_conversion(self.data)

        self.assertAlmostEqual(df.iloc[0].hourly_mean, 55)
        self.assertAlmostEqual(df.iloc[1].hourly_mean, 20530)


class AnglesToGeographicTestCase(unittest.TestCase):
    """Set up the test case for conversion from D and H to X and Y"""
    def setUp(self):
        """Create a dataframe containing mock input"""
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
        """Verify correct conversion from mock data"""
        df = io.angles_to_geographic(self.data)

        self.assertAlmostEqual(df.iloc[0].X, 11775.524238286978)
        self.assertAlmostEqual(df.iloc[0].Y, 16817.191469253001)


class WDCXYZTestCase(unittest.TestCase):
    """Set up the test case for conversion to X, Y and Z components"""
    def setUp(self):
        """Create a dataframe containing mock input"""
        self.data = pd.DataFrame(
            index=[0, 1, 2, 3, 4, 5], columns=[
                'date', 'component', 'base', 'hourly_mean_temp'])
        self.data.component = ['H', 'D', 'X', 'Y', 'Z', 'X']
        self.data.base = [200, 53, np.nan, np.nan, 300, 9999]
        self.data.hourly_mean_temp = [530, 1200, np.nan, np.nan, 430, 9999]
        self.data.date.iloc[0:5] = dt.date(day=15, month=1, year=1963)
        self.data.date.iloc[5] = dt.date(day=20, month=1, year=1963)

    def test_wdc_xyz(self):
        """Verify correct conversion to X, Y and Z components from mock data"""
        df = io.wdc_xyz(self.data)

        self.assertAlmostEqual(df.iloc[0].X, 11775.524238286978)
        self.assertAlmostEqual(df.iloc[0].Y, 16817.191469253001)
        self.assertAlmostEqual(df.iloc[0].Z, 30430.000000000000)
        self.assertTrue(np.isnan(df.iloc[1].X))

    def test_wdc_xyz_is_nan_if_Z_missing(self):
        """Ensure missing Z data are set to NaN"""
        self.data = self.data[self.data.component != 'Z']
        df = io.wdc_xyz(self.data)
        self.assertTrue(np.isnan(df.iloc[1].Z))

    def test_wdc_xyz_is_nan_if_DHXY_missing(self):
        """Ensure missing X, Y, D or H data are set to NaN"""
        self.data = self.data[~(self.data.component.isin(['D', 'H', 'X',
                                                          'Y']))]
        df = io.wdc_xyz(self.data)
        self.assertTrue(np.isnan(df.iloc[0].X))
        self.assertTrue(np.isnan(df.iloc[0].Y))


class WDCReadTestCase(unittest.TestCase):
    """Set up the test case for reading WDC files"""
    def setUp(self):
        """Create a dataframe containing mock input and specify test file"""
        self.data = pd.DataFrame(columns=['date', 'X', 'Y', 'Z'])
        self.data.columns.name = 'component'
        self.data.date = pd.date_range('1883-01-01 00:30:00', freq='H',
                                       periods=5)
        self.data.X = [np.nan, 18656.736886, 18657.537749, 18660.729592,
                       18658.976990]
        self.data.Y = [np.nan, -5487.438180, -5491.801722, -5480.946278,
                       -5493.994785]
        self.data.Z = [np.nan, np.nan, np.nan, np.nan, np.nan]

        self.filename = os.path.join(TEST_DATA_PATH, 'testdata3.wdc')

    def test_wdc_readfile(self):
        """Verify correct reading of test file by comparing with mock data"""
        df = io.wdc_readfile(self.filename)

        assert_frame_equal(df.head(), self.data)


class WDCAppendTestCase(unittest.TestCase):
    """Set up the test case for appending WDC data"""
    def setUp(self):
        """Create a dataframe containing mock input and specify test file"""
        # the type matches when this is a single value range...
        self.value1 = [pd.date_range('1911-1-1 0:30', '1911-1-1 0:30')]
        self.value2 = [45294.0]
        self.dimensions = (1416, 4)
        self.filename = 'testappenddata'

    def test_append_wdc_data(self):
        """Verify correct append of test file by comparing with mock data"""
        df = io.append_wdc_data(obs_name=self.filename, path=TEST_DATA_PATH)

        self.assertEqual(self.dimensions, df.shape)
        self.assertEqual(self.value1, df['date'].head(1).values)
        self.assertAlmostEqual(self.value2, df['Z'].tail(1).values)


class WDCHourlyToCSVTestCase(unittest.TestCase):
    """
    Mock the calls I don't want to actually make or are tested elsewhere, and
    beware that patches are applied in _reverse_ order, because whoever wrote
    mock is the devil
    """

    def setUp(self):
        """Specify mock filenames and paths"""
        self.wdc_path = 'data'
        self.write_dir = './a-test-path/to-nowhere'
        self.obs_list = ['ESK']
        self.print_obs = True
        self.wdc_data = pd.DataFrame()

    @mock.patch('magpysv.io.write_csv_data')
    @mock.patch('magpysv.io.append_wdc_data')
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists', return_value=False)
    def test_wdc_to_hourly_csv_path_exists(self, mock_exists, mock_makedirs,
                                           mock_append_wdc_data,
                                           mock_write_csv_data):
        """Check is paths/files exist"""
        io.wdc_to_hourly_csv(wdc_path=self.wdc_path,
                             write_dir=self.write_dir,
                             obs_list=self.obs_list,
                             print_obs=self.print_obs)
        assert mock_makedirs.call_count == 1
        mock_makedirs.assert_called_with(self.write_dir)

    @mock.patch('magpysv.io.write_csv_data')
    @mock.patch('magpysv.io.append_wdc_data')
    @mock.patch('sys.stdout', new_callable=StringIO)
    @mock.patch('os.path.exists', return_value=True)
    def test_wdc_to_hourly_csv_call_print(self, mock_exists, mock_print,
                                          mock_append_wdc_data,
                                          mock_write_csv_data):
        """Check if print function called"""
        io.wdc_to_hourly_csv(wdc_path=self.wdc_path,
                             write_dir=self.write_dir,
                             obs_list=self.obs_list,
                             print_obs=self.print_obs)
        self.assertEqual(self.obs_list[0] + '\n', mock_print.getvalue())

    @mock.patch('magpysv.io.write_csv_data')
    @mock.patch('magpysv.io.append_wdc_data')
    @mock.patch('os.path.exists', return_value=True)
    def test_wdc_to_hourly_csv_call_write(self, mock_exists,
                                          mock_append_wdc_data,
                                          mock_write_csv_data):
        """Check if write function called"""
        mock_append_wdc_data.return_value = self.wdc_data
        io.wdc_to_hourly_csv(wdc_path=self.wdc_path,
                             write_dir=self.write_dir,
                             obs_list=self.obs_list,
                             print_obs=self.print_obs)
        mock_write_csv_data.assert_called_with(data=self.wdc_data,
                                               write_dir=self.write_dir,
                                               obs_name=self.obs_list[0])
