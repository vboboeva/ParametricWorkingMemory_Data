import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
import pandas
from pandas import DataFrame
import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import curve_fit
import color_palette as cp
# from helper_func import func
# from helper_make_data import make_data
# from helper_mymax import mymax
# from helper_history import history

def history(trialsback, stimuli, readout,
	stimulus_set, num_stimpairs,
	ids=None
	):

	if ids is None:
		ids = np.arange(len(stimuli))

	ids_shifted = ids - trialsback
	_mask = ids_shifted > -1
	ids = ids[_mask]
	ids_shifted = ids_shifted[_mask]

	stims = stimuli[ids]
	stims_shifted = stimuli[ids_shifted]

	trialtypevals=np.zeros((len(stimulus_set), len(stimulus_set)))
	responsevals=np.zeros((len(stimulus_set), len(stimulus_set)))

	#print(np.arange(0, num_sims*num_trials, num_trials))
	# SORT performance by previous pair of stimuli

	for idx, stim, stim_s in zip(ids, stims, stims_shifted):
		for m in range(len(stimulus_set)):
			if ( stim == stimulus_set[m] ).all():
				for n in range(len(stimulus_set)):
					if ( stim_s == stimulus_set[n] ).all():
						trialtypevals[n,m] += 1
						responsevals[n,m] += readout[idx]

	x=int(num_stimpairs/2)
	# A11=np.divide(responsevals[:x,:x], trialtypevals[:x,:x], out=np.zeros_like(responsevals[:x,:x]), where=trialtypevals[:x,:x]!=0) 
	# B11=np.zeros((x,x))
	# for i in range(x):
	# 	B11[:,i] = A11[:,i] - np.mean(A11[:,i])
	A11=np.divide(responsevals[x:,x:], trialtypevals[x:,x:], out=np.zeros_like(responsevals[x:,x:]), where=trialtypevals[x:,x:]!=0)
	A12=np.divide(responsevals[x:,:x], trialtypevals[x:,:x], out=np.zeros_like(responsevals[x:,:x]), where=trialtypevals[x:,:x]!=0)
	A21=np.divide(responsevals[:x,x:], trialtypevals[:x,x:], out=np.zeros_like(responsevals[:x,x:]), where=trialtypevals[:x,x:]!=0) 
	A22=np.divide(responsevals[:x,:x], trialtypevals[:x,:x], out=np.zeros_like(responsevals[:x,:x]), where=trialtypevals[:x,:x]!=0)
	R = responsevals[x:,x:] + responsevals[x:,:x] + responsevals[:x,x:] + responsevals[:x,:x]
	T = trialtypevals[x:,x:] + trialtypevals[x:,:x] + trialtypevals[:x,x:] + trialtypevals[:x,:x]
	A = np.divide(R, T, out=np.zeros_like(R), where=T!=0) 
	# Set to chance those values for which we don't have any trials at all
	# this is mainly for bimodal_l2
	A11[np.where(A11 == 0.)]=np.nanmean(A11)
	A12[np.where(A12 == 0.)]=np.nanmean(A12)
	A21[np.where(A21 == 0.)]=np.nanmean(A21)
	A22[np.where(A22 == 0.)]=np.nanmean(A22)

	B11=np.zeros((x,x))
	B12=np.zeros((x,x))
	B21=np.zeros((x,x))
	B22=np.zeros((x,x))
	B=np.zeros((x, x))

	for i in range(x):
		B11[:,i] = A11[:,i] - np.nanmean(A11[:,i])
		B12[:,i] = A12[:,i] - np.nanmean(A12[:,i])
		B21[:,i] = A21[:,i] - np.nanmean(A21[:,i])
		B22[:,i] = A22[:,i] - np.nanmean(A22[:,i])
		B[:,i] = A[:,i] - np.nanmean(A[:,i])

	H=np.divide(responsevals, trialtypevals, out=np.zeros_like(responsevals), where=trialtypevals!=0)
	# Set to chance those values for which we don't have any trials at all
	# H[np.where(H == 0.)]=0.5
	# print('H', H[x:,x:])
	return B11, B12, B21, B22, B, H
