# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:06:42 2019

@author: jefferson
"""
import pandas as pd

base = pd.read_csv('databases/census.csv')
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
labelenconder_previsores = LabelEncoder()
    from sklearn.cross_validation import train_test_split


previsores[:,1] = labelenconder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelenconder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelenconder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelenconder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelenconder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelenconder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelenconder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelenconder_previsores.fit_transform(previsores[:,13])

#previsores = base.iloc[:,8:9].values
#previsores[:,0] = labelenconder_previsores.fit_transform(previsores[:,0])

onehotenconder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotenconder.fit_transform(previsores).toarray()
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.2, random_state=0)