#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:46:31 2018

@author: Tao Su collected.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics.pairwise import paired_distances 
import hdbscan

#ClusterClassifier Framework, requires fit and predict method.
class ClusterClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clustering_model=None):
        self.clustering_model = clustering_model
        self.comparison_summary=None


    def fit(self, attributes0, label0):      
        attributes=np.array(attributes0)
        label=np.array(label0)
        
        self.clustering_model.fit(attributes)
        pred=self.clustering_model.predict(attributes)

        lab=pd.Series(label,name='lab',dtype=int)
        pre=pd.Series(pred,name='pre',dtype=int)
        kmcomp=pd.concat([lab,pre],axis=1)
        
        kmc1=kmcomp[kmcomp['lab']==1]
        kmc0=kmcomp[kmcomp['lab']==0]
        
        sta1=kmc1['pre'].groupby(kmc1['pre']).count()
        sta0=kmc0['pre'].groupby(kmc0['pre']).count()        
        
        align=pd.concat([sta1/len(kmc1),sta0/len(kmc0)],axis=1).fillna(0)
        align.columns=['sta1a','sta0a']
        
        sta1a=align['sta1a']
        sta0a=align['sta0a']
        
        dif=sta1a/(sta1a+sta0a)
        result=pd.concat([pd.concat([sta1,sta0,sta1a,sta0a],axis=1).fillna(0),
                          dif],axis=1)
        result.columns=['sta1','sta0','sta1a','sta0a','dif']
        resee=result.sort_values('dif',ascending=False)
        resee['cumsta1a']=np.cumsum(resee['sta1a'])
        resee['cumsta0a']=np.cumsum(resee['sta0a'])
        resee['cumsta1']=np.cumsum(resee['sta1'])
        resee['cumsta0']=np.cumsum(resee['sta0'])        
      
        self.comparison_summary=resee
        
        
class KmeansClusterClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clustering_model=None):
        self.clustering_model = clustering_model
        self.comparison_summary=None


    def fit(self, attributes0, label0):      
        attributes=np.array(attributes0)
        label=np.array(label0)
        
        self.clustering_model.fit(attributes)
        pred=self.clustering_model.predict(attributes)

        lab=pd.Series(label,name='lab',dtype=int)
        pre=pd.Series(pred,name='pre',dtype=int)
        kmcomp=pd.concat([lab,pre],axis=1)
        
        kmc1=kmcomp[kmcomp['lab']==1]
        kmc0=kmcomp[kmcomp['lab']==0]
        
        sta1=kmc1['pre'].groupby(kmc1['pre']).count()
        sta0=kmc0['pre'].groupby(kmc0['pre']).count()        

        align=pd.concat([sta1/len(kmc1),sta0/len(kmc0)],axis=1).fillna(0)
        align.columns=['sta1a','sta0a']
        
        sta1a=align['sta1a']
        sta0a=align['sta0a']
        
        dif=sta1a/(sta1a+sta0a)
        result=pd.concat([pd.concat([sta1,sta0,sta1a,sta0a],axis=1).fillna(0),
                          dif],axis=1)
        result.columns=['sta1','sta0','sta1a','sta0a','dif']
        resee=result.sort_values('dif',ascending=False)
        resee['cumsta1a']=np.cumsum(resee['sta1a'])
        resee['cumsta0a']=np.cumsum(resee['sta0a'])
        resee['cumsta1']=np.cumsum(resee['sta1'])
        resee['cumsta0']=np.cumsum(resee['sta0'])        

        kmcomp['dis']=pd.Series(map(lambda x1,x2: paired_distances(x1.reshape(1,-1),
           self.clustering_model.cluster_centers_[x2].reshape(1,-1))[0],attributes,pred))
    
        distance_max=kmcomp['dis'].groupby(kmcomp['pre']).max().rename('distance_max')
        distance_mean=kmcomp['dis'].groupby(kmcomp['pre']).mean().rename('distance_mean')       
        distances=pd.concat([distance_max,distance_mean],axis=1)
        resee=pd.merge(resee,pd.DataFrame(distances),left_index=True, right_index=True)
        
        self.comparison_summary=resee


        
    def predict(self,attributes0,thresh=0.75,compu_dis=False):
        
        attributes=np.array(attributes0)

        pick_clusters=self.comparison_summary[self.comparison_summary['dif']>=thresh].index.tolist()
        
        res=self.clustering_model.predict(attributes)
        ret=pd.Series(res).isin(pick_clusters).astype(int).values
        
        if compu_dis:
            kmcdis=pd.Series(map(lambda x1,x2: paired_distances(x1.reshape(1,-1),
               self.clustering_model.cluster_centers_[x2].reshape(1,-1))[0],attributes,res)) 
            return ret,kmcdis
        else:
            kmcdis=None
            return ret
        
        
class HDBSCANClusterClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clustering_model=None):
        self.clustering_model = clustering_model
        self.comparison_summary=None


    def fit(self, attributes0, label0):      
        attributes=np.array(attributes0)
        label=np.array(label0)
        
        self.clustering_model.fit(attributes)
        pred=self.clustering_model.approximate_predict(self.clustering_model, attributes)

        lab=pd.Series(label,name='lab',dtype=int)
        pre=pd.Series(pred,name='pre',dtype=int)
        kmcomp=pd.concat([lab,pre],axis=1)
        
        kmc1=kmcomp[kmcomp['lab']==1]
        kmc0=kmcomp[kmcomp['lab']==0]
        
        sta1=kmc1['pre'].groupby(kmc1['pre']).count()
        sta0=kmc0['pre'].groupby(kmc0['pre']).count()        
        
        align=pd.concat([sta1/len(kmc1),sta0/len(kmc0)],axis=1).fillna(0)
        align.columns=['sta1a','sta0a']
        
        sta1a=align['sta1a']
        sta0a=align['sta0a']
        
        dif=sta1a/(sta1a+sta0a)
        result=pd.concat([pd.concat([sta1,sta0,sta1a,sta0a],axis=1).fillna(0),
                          dif],axis=1)
        result.columns=['sta1','sta0','sta1a','sta0a','dif']
        resee=result.sort_values('dif',ascending=False)
        resee['cumsta1a']=np.cumsum(resee['sta1a'])
        resee['cumsta0a']=np.cumsum(resee['sta0a'])
        resee['cumsta1']=np.cumsum(resee['sta1'])
        resee['cumsta0']=np.cumsum(resee['sta0'])        
      
        self.comparison_summary=resee


        
    def predict(self,attributes0,thresh=0.75):
        
        attributes=np.array(attributes0)

        pick_clusters=self.comparison_summary[self.comparison_summary['dif']>=thresh].index.tolist()
        
        res,cluster_strengths=self.clustering_model.approximate_predict(self.clustering_model, attributes)
        
        return (pd.Series(res).isin(pick_clusters).astype(int).values,cluster_strengths)
        
      
        
#The codes below are from the github of the book:
#Hands-on machine learning with sklearn and tensorflow.

# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
