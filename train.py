from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification

import pickle

from sklearn.ensemble import RandomForestClassifier

# Load Data

df = pd.read_csv("credit_risk_dataset.csv")

# Outliers

df['person_age'] = df['person_age'].apply(lambda x: x if x <= 90 else np.nan)
df['person_income'] = df['person_income'].apply(lambda x: x if x <= 1000000 else np.nan)
df['person_emp_length'] = df['person_emp_length'].apply(lambda x: x if x <= 45 else np.nan)

# Nulls and type

df['loan_int_rate'].fillna(df.loan_int_rate.median(), inplace=True)

# target

target = 'loan_status'
features = df.columns[df.columns!=target]

######### Divide de dataframe (carrefully it is not an array for now!)
X = df[features]
y = df[target]

# Funcion

def obtener_listas_de_variables(dataset):
  num = []
  binary = []
  cat = []

  for i in dataset:
    if (dataset[i].dtype.kind == 'i' or dataset[i].dtype.kind == 'f') and (i not in target) and (len(dataset[i].unique()) != 2):
      num.append(i)
    elif (dataset[i].dtype.kind == 'i' or dataset[i].dtype.kind == 'f' or dataset[i].dtype.kind == 'O') and (i not in target) and (len(dataset[i].unique()) == 2):
      binary.append(i)
    elif (dataset[i].dtype.kind == 'O') and (i not in target):
      cat.append(i)
    else:
      print(target)

  return num, binary, cat

num, binary, cat = obtener_listas_de_variables(df)

def highly_correlated(X, y, threshold):
    col_corr = list() # Set of all the names of deleted columns
    colnames = list()
    rownames = list()
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colnames.append(corr_matrix.columns[i]) # getting the name of column
                rownames.append(corr_matrix.index[j])
                col_corr.append(corr_matrix.iloc[i, j])
    Z = pd.DataFrame({'F1':colnames,
                      'F2':rownames,
                      'corr_F1_F2':col_corr,
                      'corr_F1_target': [np.abs(np.corrcoef(X[i],y, rowvar=False)[0,1]) for i in colnames],
                      'corr_F2_target': [np.abs(np.corrcoef(X[i],y, rowvar=False)[0,1]) for i in rownames]
                      })
    Z['F_to_delete'] = rownames
    Z['F_to_delete'][Z['corr_F1_target'] < Z['corr_F2_target']] = Z['F1'][Z['corr_F1_target'] < Z['corr_F2_target']]
    
    return Z

# Preprocessing

df = pd.get_dummies(df, columns= cat)
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1}).astype('bool')
target = 'loan_status'
features = df.columns[df.columns!=target]
X = df[features]
y = df[target]
highly_corr = highly_correlated(X,y,0.95)
vt = VarianceThreshold(threshold = 0.01) #Eliminamos columnas donde el 99% de los valores son iguales
vt.fit(X)

cols_lowvar = X.columns[vt.get_support()==False]
X.drop(columns=cols_lowvar,inplace=True)

# Split Data

kfold = KFold(n_splits = 6, shuffle = True, random_state=111)
number = 1
for train_cv, test_cv in kfold.split(X, y):
    print(f'Fold:{number}, Train set: {len(train_cv)}, Test set:{len(test_cv)}')
    number += 1
train_cv, test_cv = make_classification(n_samples = 2*5276, n_classes = 2, weights = [0.99, 0.01], random_state=111)
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split (train_cv, test_cv, test_size = 0.5, random_state = 111, stratify=test_cv)

rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=10, min_samples_leaf=5)
Model_RandomForestClassifier = rf.fit(X_train_cv, y_train_cv)


with open('model.pkl','wb') as file:
    pickle.dump(Model_RandomForestClassifier,file)
    print("Model saved successfully")

