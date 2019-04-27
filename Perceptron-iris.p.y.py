# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:42:21 2019

@author: jeffe
"""
import pandas as pd

base = pd.read_csv('databases/iris.data.txt')

base.describe()

base.loc[pd.isnull(base['5.1'])]