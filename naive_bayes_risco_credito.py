# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:04:51 2020

@author: jeffe
"""

import pandas as pd

base = pd.read_csv('databases/risco-credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores,classe)