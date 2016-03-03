# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:26:22 2016

@author: Grace
"""
import unittest
import os
from wdc_io import wdc_to_dataframe


class WDCTestCase(unittest.TestCase):
    
        
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
    
    def test_wdc_to_dataframe(self):
        
        testfile = os.path.join(self.path, 'testdata.wdc')
        
        data = wdc_to_dataframe(testfile)
        # Observatory code
        self.assertEqual(data.code[0], 'NGK')
        self.assertEqual(len(data.code.unique()),1)
        # Components
        self.assertTrue(any(x in {'X', 'Y', 'Z', 'D', 'I', 'H'} for x in data.component))
        self.assertEqual(data.component[3],'D')
        self.assertEqual(data.component.values[-1],'Z')
        # Daily mean (using the tabular base)
        self.assertAlmostEqual(data.daily_mean.values[-1],45113)
        self.assertAlmostEqual(data.daily_mean[3],800)
        
        
                
class DailyToMonthlyTestCase(unittest.TestCase):
    
        
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
    
    def test_wdc_to_dataframe(self):
        
        testfile = os.path.join(self.path, 'testdaily.csv')
        
        data = wdc_to_dataframe(testfile)