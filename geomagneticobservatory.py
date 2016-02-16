# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:58:32 2016

@author: Grace
"""

class GeomagneticObservatory(object):
    
    
    def __init__(self, name=None, lat=0.0, lon=0.0, alt=0.0):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.alt = alt