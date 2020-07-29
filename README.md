# HowToCreateAPipelineML
the line of codes show how to build a pipeline in Machine Learning 

#import the necessary libraries
import sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor

#calling the data
melbourne_data = pd.read_csv(r"C:\Users\Blessing\Downloads\Uworld\melb_data.csv")

#droping some column, either they highly incomplete or have no importance to the Machine Learning prediction
house = mel_data.drop(['Suburb','Date', 'Price','SellerG', 'Address','BuildingArea','YearBuilt'], axis=1)
#house.head()

#to find total number of missing value in a column 
#mising = house.isna().sum()
#k = mising>0
#k

#asigning the target 
Y = mel_data.Price


#1st stage of the pipeline 
missing_col =['Car', 'CouncilArea']
Pipe1 = Pipeline(steps=[('impute', SimpleImputer(missing_values=np.nan,strategy='most_frequent') ), ('OHE', OneHotEncoder(handle_unknown='ignore'))])

#2BD STAGE
objects_col = ['Type', 'Method', 'Regionname', 'Rooms','Bathroom','Bedroom2']
Pipe2 = Pipeline(steps=[('OHE', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('pipe1' Pipe1, missing_col), ('pipe2', Pipe2, objects_col)], remainder='passthrough')


#you can use different learning algorithmn by changing the regressor, maybe from XGBRegressir to RandomForestRegressor  
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor())])

model.fit(housing, Y)


#on the test dataset you can make prediction by calling
model.predict()
