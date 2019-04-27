# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:56:31 2019

@author: jeffe
"""

import pandas as pd

base = pd.read_csv('databases/credit-data.csv')
base.describe()
base.loc[base['age'] < 0]
#apagar a coluna
#base.drop('age',1,inplace=True)
#apagar somente os registros com problema
base.drop(base[base['age'] < 0].index,inplace=True)
#preencher os valores manualmente
#preencher os valores com a media
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]  

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.2, random_state=0)