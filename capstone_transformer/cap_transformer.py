import pandas as pd
import numpy as np
import io
import os
import json
import re
import math
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

with open(os.path.join("data", "lat_dict.json")) as json_file:
    lat_dict = json.load(json_file)

with open(os.path.join("data", "long_dict.json")) as json_file:
    long_dict = json.load(json_file) 


def fixing_longitude(row):
    '''
    For an unknown Longitude, returns average Latitude of station in Longitude
    '''

    ans = row['Longitude']
    if math.isnan(row['Longitude']):
        stat_i = row['station']
        ans = long_dict[stat_i]
    return ans

def date_to_datetime(data):
    data['Date'] = pd.to_datetime(data['Date'],infer_datetime_format=True)
    return data

def fixing_latitude(row):
    '''
    For an unknown Latitude, returns average Latitude of station in dataset
    '''

    ans = row['Latitude']
    if math.isnan(row['Latitude']):
        stat_i = row['station']
        ans = lat_dict[stat_i]
    return ans

def uniformize_text (text):
    '''
    Removes punctuation and lowers case
    '''
    pattern = r"[^\w\d\s]"
    text_no_p = re.sub(pattern,"",text)

    return text_no_p.lower()


class DateTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self, X,y=None):
        X_=X.copy()
        X_['Date'] = pd.to_datetime(X_['Date'],infer_datetime_format=True)
        #needed for RFC
        X_['Month'] = X_['Date'].apply(lambda x:x.month)
        X_['Day'] = X_['Date'].apply(lambda x:x.day)
        X_['Year'] = X_['Date'].apply(lambda x:x.year)
        X_ = X_.drop(columns=['Date'])

        return X_
    
class PartPolicingTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self, X,y=None):
        X_=X.copy()
        X_['Part of a policing operation'] = X_['Part of a policing operation'].astype(bool)
        return X_

class CoordTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X,y=None):
        X_=X.copy()
        X_['Longitude'] = X_.apply(fixing_longitude, axis = 1)
        X_['Latitude'] = X_.apply(fixing_latitude, axis = 1)

        return X_
    
class CatProcTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):

        pass
    
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X,y=None):
        X_=X.copy() 
        X_=X_.astype(str)
        X_= X_.apply(uniformize_text)


        return X_

class CapTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, features, num_feats, cat_feats):
        self.features=features
        self.num_feats=num_feats
        self.cat_feats=cat_feats
    
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X,y=None):
        X_=X.copy()
        
        
        for column in X_.columns:
            
            if column in self.num_feats:
                X_[column]=pd.to_numeric(X_[column],errors='coerce')


            if column in self.cat_feats:
                if column != 'Part of a policing operation':
                    X_[column]=X_[column].astype(str)
                    X_[column] = X_[column].apply(uniformize_text)
                elif column == 'Part of a policing operation':
                    X_[column] = X_[column].astype(bool)
            
            if column == 'Longitude':
                X_['Longitude'] = X_.apply(fixing_longitude, axis = 1)
            if column == 'Latitude':
                X_['Latitude'] = X_.apply(fixing_latitude, axis = 1)
        
        X_=date_to_datetime(X_)
        
        
        fts = self.features
        
        X_=X_[fts]


        return X_