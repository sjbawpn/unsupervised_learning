import numpy as np
import matplotlib.pyplot as plt
import ann
import decomposition
from decomposition import pca_decompose as pca
from decomposition import lda_decompose as lda
import common
from common import Bio
from common import Pen
from common import scale

def run_ann(dataset):
    scores1 = ann.run(dataset)
    #print scores
    #common.plot("Learner Accuracy over epochs", "epochs", "Accuracy", range(1,21), scores, 
    #    "momentum={0}, hidden data={1}, weight decay={2}".format(0.1, 10, 0.01))

    dataset_scaled = scale(dataset)
    scores2 = ann.run(dataset_scaled)

    dataset_pca = pca(dataset, dataset.all.features.shape[1]/2)
    scores3 = ann.run(dataset_pca)

    plt.title("Learner Accuracy over epochs")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,21), scores1, label="original")
    plt.plot(range(1,21), scores2, label="scaled")
    plt.plot(range(1,21), scores3, label="pca decomposed")
    plt.legend()

dataset = Bio()
dataset_lda = lda(dataset, 5)