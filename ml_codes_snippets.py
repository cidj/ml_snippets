#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:46:31 2018

@author: Tao Su
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
        self.ordered_centers=None


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
        self.ordered_centers=pd.DataFrame(self.clustering_model.cluster_centers_).iloc[resee.index]



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

        pred=self.clustering_model.labels_

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

        res,cluster_strengths=hdbscan.approximate_predict(self.clustering_model, attributes)

        return (pd.Series(res).isin(pick_clusters).astype(int).values,cluster_strengths)


# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet.
# From book Hands-on machine learning with sklearn and tensorflow
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


class DataFrameImputer(BaseEstimator,TransformerMixin):

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


# From book Hands-on machine learning with sklearn and tensorflow.
# It assumes that the importances have already been computed.
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


# Use a selector to do features select, which may slow down coomputations..
class FeatureSelector(BaseEstimator,TransformerMixin):

    def __init__(self,num,selector):
        """Select features.

        Args:
            num: Number of features to selected.
            selector: A classifier can be used to fit to dataset and has
            feature_importances_ attribute.
        """
        self.num=num
        self.selector=selector
        self.features_order=None

    def fit(self, X, y=None):
        self.selector.fit(X,y)
        self.features_order=pd.Series(
                self.selector.feature_importances_,
                index=X.columns).sort_values(ascending=False)
        return self

    def transform(self, X, y=None):
        return X[self.features_order.index[0:self.num].tolist()]


# A simple Keras Classifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier

def append_dropout_layer(model, neurons_per_layer, dropout_ratio):
    model.add(Dense(neurons_per_layer,activation='relu',
                    kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout_ratio))

def make_model(number_of_inputs,number_of_layers=2, neurons_per_layer=32, dropout_ratio=0.2, optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_shape=[number_of_inputs],
                    activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout_ratio))
    for i in range(number_of_layers-1):
        append_dropout_layer(model, neurons_per_layer, dropout_ratio)
    model.add(Dense(1,  activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model

# An example:
# km=KerasClassifier(build_fn=make_model,number_of_inputs=10,
#             nb_epoch=100,batch_size=128,dropout_ratio=0.1,verbose=0)


class TreesDataFrameImputer(BaseEstimator):

    def __init__(self):
        """Impute missing values. Only dtype object are acceptable

        If transform method is used on new data, some columns are imputed with
        mean values, where there are not nan values in the fit data.
        values.

        This imputer cannot be used in pipeline, which requires a transform
        mechod that only has one parameter.

        """

    def fit(self, X, y=None):

        from sklearn.tree import DecisionTreeRegressor

        X_perc=X.count()/X.shape[0]

        self.cols_ful=X_perc[X_perc==1].index.tolist()
        self.cols_nan=X_perc[X_perc!=1].index.tolist()

        self.means = X[self.cols_ful].mean().rename('means')

        Xy=pd.concat([X[self.cols_ful],y],axis=1)

        tY=X[self.cols_nan]
        self.reg_dict=dict()

        for i in self.cols_nan:
            ty=tY[i]

            X_train=Xy[ty.notnull()]
            ty_train=ty[ty.notnull()]

            rf=DecisionTreeRegressor()
            rf.fit(X_train,ty_train)
            self.reg_dict[i]=rf

        return self

    def transform(self, X, y=None):

        X_ful=X[self.cols_ful].fillna(self.means)

        Xy=pd.concat([X_ful,y],axis=1)

        ty_ls=[]
        tY=X[self.cols_nan]
        for i in self.cols_nan:
            ty=tY[i]

            if ty.isnull().sum()>0:
                X_test=Xy[ty.isnull()]
                ty_test=ty[ty.isnull()]

                ty_predict=pd.Series(
                        self.reg_dict[i].predict(X_test),
                        index=ty_test.index,
                        name=ty_test.name,
                        dtype=ty_test.dtype)
                ty_ls.append(ty_predict)

        to_fill=pd.concat(ty_ls,axis=1)
        return X.fillna(self.means).fillna(to_fill)


def TreesImpute(X,y):

    from sklearn.tree import DecisionTreeRegressor

    X_perc=X.count()/X.shape[0]

    cols_ful=X_perc[X_perc==1].index
    cols_nan=X_perc[X_perc!=1].index

    Xy=pd.concat([X[cols_ful],y],axis=1)

    tY=X[cols_nan]
    ty_ls=[]

    for i in cols_nan:
        ty=tY[i]

        X_train=Xy[ty.notnull()]
        ty_train=ty[ty.notnull()]
        X_test=Xy[ty.isnull()]
        ty_test=ty[ty.isnull()]

        rf=DecisionTreeRegressor()
        rf.fit(X_train,ty_train)

        ty_predict=pd.Series(
                rf.predict(X_test),
                index=ty_test.index,
                name=ty_test.name,
                dtype=ty_test.dtype)
        ty_ls.append(ty_predict)

    to_fill=pd.concat(ty_ls,axis=1)

    return X.fillna(to_fill)
