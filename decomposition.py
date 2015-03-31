import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.mixture import GMM 
from sklearn.decomposition import PCA 
from sklearn.decomposition import FastICA 
from sklearn.decomposition import RandomizedPCA 
from sklearn.lda import LDA 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
import common
from common import DataSet
from common import Data

# Input DataSet
def pca_decompose(dataset,n):
    pca = PCA(n_components=n)
    reduced_features = pca.fit_transform(dataset.all.features)
    training_size = dataset.training_size
    training = Data(reduced_features[:training_size,:], dataset.all.target[:training_size])
    testing = Data(reduced_features[training_size:,:], dataset.all.target[training_size:])
    return DataSet(training, testing)

def ica_decompose(dataset,n):
    ica = FastICA(n_components=n)
    reduced_features = ica.fit_transform(dataset.all.features)
    training_size = dataset.training_size
    training = Data(reduced_features[:training_size,:], dataset.all.target[:training_size])
    testing = Data(reduced_features[training_size:,:], dataset.all.target[training_size:])
    return DataSet(training, testing)

def rca1_decompose(dataset,n):
    rca = RandomizedPCA(n_components=n)
    reduced_features = rca.fit_transform(dataset.all.features)
    training_size = dataset.training_size
    training = Data(reduced_features[:training_size,:], dataset.all.target[:training_size])
    testing = Data(reduced_features[training_size:,:], dataset.all.target[training_size:])
    return DataSet(training, testing)

def rca2_decompose(dataset,n):
    rca = GaussianRandomProjection(n_components=n)
    reduced_features = rca.fit_transform(dataset.all.features)
    training_size = dataset.training_size
    training = Data(reduced_features[:training_size,:], dataset.all.target[:training_size])
    testing = Data(reduced_features[training_size:,:], dataset.all.target[training_size:])
    return DataSet(training, testing)

def lda_decompose(dataset, n): 
    lda = LDA(n_components=n)
    reduced_features = lda.fit_transform(dataset.all.features, dataset.all.target)
    training_size = dataset.training_size
    training = Data(reduced_features[:training_size,:], dataset.all.target[:training_size])
    testing = Data(reduced_features[training_size:,:], dataset.all.target[training_size:])
    return DataSet(training, testing)