# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:59:42 2016

@author: Grace
"""

class FieldModel(object):
    
    
    def __init__(self, name=None, start=None, end=None, resolution=14):
        self.name = name
        self.start = start
        self.end = end
        self.resolution = resolution