import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
import numpy as np
import time
from sklearn.preprocessing import scale as skscale

class Data(object):
    def __init__(self, features, target):
        self.features = features
        self.target = target

class DataSet(object):
    def __init__(self, training, testing):       
        features = np.vstack([training.features, testing.features])
        target = np.hstack([training.target, testing.target])
        training_size = training.features.shape[0]
        self.training_size = training_size
        self.training = Data(features[:training_size,:], target[:training_size])
        self.testing = Data(features[training_size:,:], target[training_size:])
        self.all = Data(features,target)



"""
Bio response
"""
def Bio():
    bio_data = np.loadtxt("datasets/bio/train.csv", delimiter=',');
    train, test= train_test_split(bio_data, test_size=0.25)

    features = train[:,1:]
    target = train[:,0]
    bio_train = Data(features,target)

    test_features = test[:,1:]
    test_target = test[:,0]
    bio_test = Data(test_features, test_target)

    return DataSet(bio_train, bio_test)

"""
Cancer
"""
def Cancer():
    cancer_data = np.loadtxt("datasets/cancer/wdbc.data", delimiter=',');
    cancer_data = cancer_data[:,1:]
    train, test= train_test_split(cancer_data, test_size=0.25)

    features = train[:,1:]
    target = train[:,0]
    cancer_train = Data(features,target)

    test_features = test[:,1:]
    test_target = test[:,0]
    cancer_test = Data(test_features, test_target)

    return DataSet(cancer_train, cancer_test)

"""
Sat Log
"""
def Sat():
    sat_tra = np.loadtxt("datasets/satlog/sat.trn", delimiter=' ');
    features = sat_tra[:,:-1]
    target = sat_tra[:,-1]
    satlog_train = Data(features,target)

    sat_tst = np.loadtxt("datasets/satlog/sat.tst", delimiter=' ');
    test_features = sat_tst[:,:-1]
    test_target = sat_tst[:,-1]
    satlog_test = Data(test_features, test_target)

    return DataSet(satlog_train, satlog_test)

""" 
Pen Data
"""
def Pen():
    pen_tra = np.loadtxt("datasets/pendigits/pendigits.tra", delimiter=',');
    features = pen_tra[:,:-1]
    target = pen_tra[:,-1]
    pen_train = Data(features,target)

    pen_tst = np.loadtxt("datasets/pendigits/pendigits.txt", delimiter=',');
    test_features = pen_tst[:,:-1]
    test_target = pen_tst[:,-1]
    pen_test = Data(test_features, test_target)

    return DataSet(pen_train, pen_test)

"""
Poker Data
"""
def Poker():
    poker_data = np.loadtxt("datasets/poker/train.csv", delimiter=',');
    train, test= train_test_split(poker_data, test_size=0.25)

    features = train[:,:-1]
    target = train[:,-1]
    poker_train = Data(features,target)

    test_features = test[:,:-1]
    test_target = test[:,-1]
    poker_test = Data(test_features, test_target)

    return DataSet(poker_train, poker_test)


default = Poker()

"""

Scale the DataSet
"""
def scale(dataset, with_mean=True):
    scaled_features = skscale(dataset.all.features, with_mean=with_mean)
    training_size = dataset.training_size
    training = Data(scaled_features[:training_size,:], dataset.all.target[:training_size])
    testing = Data(scaled_features[training_size:,:], dataset.all.target[training_size:])
    return DataSet(training, testing)

def trim_attributes(estimator, dataset):
    estimator = estimator.fit(dataset.all.features, dataset.all.target)
    dataset.training.features = estimator.transform(dataset.training.features)
    #dataset.training.target = estimator.transform(dataset.training.target)
    dataset.testing.features = estimator.transform(dataset.testing.features)
    dataset.all.features = estimator.transform(dataset.all.features)
    #dataset.data.target = estimator.transform(dataset.data.target)
    return dataset

def score_learner(estimator, dataset=default):
    estimator = estimator.fit(dataset.training.features, dataset.training.target)
    score = estimator.score(dataset.testing.features, dataset.testing.target);
    return score

def plot_learning_curve(estimator, title, dataset=default, n_iter=100, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    X = dataset.all.features
    y = dataset.all.target
    if cv == None:
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=n_iter,
                                   test_size=0.2, random_state=0)  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot(title, xlabel, ylabel, xdata, ydata, legend, fig = None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xdata, ydata, label=legend)
    plt.legend()