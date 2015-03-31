import common
from common import Bio
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = Bio()
#nb_classes= len(np.unique(dataset.training.target))

def run(dataset, hidden=10, mom=0.1, wd=0.01, epochs=20):
	nb_classes = max(dataset.training.target) + int(min(dataset.training.target) == 0)

	#xr = np.arange(0.1,1.1,0.1)
	trndata = ClassificationDataSet(len(dataset.training.features[0]), 1, nb_classes=nb_classes)
	for n in xrange(len(dataset.training.features)):
		trndata.addSample(dataset.training.features[n], [dataset.training.target[n]])

	trndata._convertToOneOfMany( )

	print "Number of training patterns: ", len(trndata)
	print "Input and output dimensions: ", trndata.indim, trndata.outdim
	print "First sample (input, target, class):"

	

	fnn = buildNetwork( trndata.indim, hidden, trndata.outdim)
	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=mom, verbose=False, weightdecay=wd)
	scores = []
	for i in range(epochs):
		trainer.trainEpochs( 1 )
		ypreds = []
		ytrues = []
		for i in range(dataset.training.features.shape[0]):
		    pred = fnn.activate(dataset.training.features[i, :])
		    ypreds.append(pred.argmax())
		    ytrues.append(dataset.training.target[i])
		score = accuracy_score(ytrues, ypreds, normalize=True)
		scores.append(score)
		print "Accuracy on test set: {0}".format(score)
	
	return scores