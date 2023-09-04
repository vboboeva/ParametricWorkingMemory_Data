import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
# import sklearn.linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron, LogisticRegression
import pandas
from pandas import DataFrame
# import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import curve_fit
# import color_palette as cp

def compute_performance(stimulus_set, stimuli, readout, labels):
	# SORT performance by pair of stimuli (for performance by stimulus type)
	performvals=np.zeros(len(stimulus_set))
	for m in range(len(stimulus_set)):
		l=stimulus_set[m]
		indices=np.where(np.all(stimuli==l, axis=1))[0]
		if (len(indices) != 0):
			performvals[m]=len(np.where(readout[indices] == labels[indices])[0])/len(labels[indices])

	return performvals