# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Harsh/Downloads/diseases/trained_model.sav', 'rb'))
diabetes_dataset=pd.read_csv('C:/Users/Harsh/Downloads/diseases/diabetes.csv')
X=diabetes_dataset.drop(columns='Outcome',axis=1)
scaler=StandardScaler()
scaler.fit(X)

input_data = (1,85,66,29,0,26.6,0.351,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
prediction = loaded_model.predict(std_data)

print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')