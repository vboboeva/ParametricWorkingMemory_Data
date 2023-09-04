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
import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import curve_fit
from helper_make_data import make_data
# from helper_mymax import mymax
from helper_history import history
from helper_func import func
import sys
import color_palette as cp

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
# rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
# rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

def plot_B(B, trialsbacks, label, fig, axs, stimulus_set, color):
	num_stimpairs=np.shape(stimulus_set)[0]

	for i, o in enumerate(range(1, trialsbacks)):

		# plot the mean
		xdata=np.arange(int(num_stimpairs/2))
		bias=np.mean(B[:,:,i]*100, axis=1)
		popt, pcov = curve_fit(func, xdata, bias)
		axs[i].scatter(xdata, bias, s=5, color=color)
		axs[i].plot(xdata, func(xdata, popt[0], popt[1]), label=label, color=color)

		# plot individual curves
		# for j in range(int(num_stimpairs/2)):
		# 	ydata=B[:,j,i]*100
		# 	# axs[i].scatter(xdata, ydata, alpha=0.5, s=5)
		# 	popt, pcov = curve_fit(func, xdata, ydata)
		# 	axs[i].plot(xdata, func(xdata, popt[0], popt[1]), alpha=0.3)

		axs[i].set_xticks(np.arange(0,5)) 
		axs[i].set_xticklabels(['%.1f,%.1f'%(stimulus_set[j,0], stimulus_set[j,1] ) for j in range(5) ],  rotation=45)
		axs[i].spines['right'].set_visible(False)
		axs[i].spines['top'].set_visible(False)
		axs[i].set_ylim(-25,25)

		if i==0:
			axs[i].set_ylabel("Stim 1 $<$ Stim 2 bias ($\%$)")
			axs[i].set_xlabel("%d trial back pair $(s_1, s_2)$"%o)
			axs[i].legend()
		else:
			axs[i].set_xticklabels([])
			axs[i].set_yticklabels([])
			axs[i].set_xlabel("%d trials back"%o)

whichsubject=str(sys.argv[1])
whichdistrib=str(sys.argv[2])
whichspecies=str(sys.argv[3])
whichdelay=str(sys.argv[4])
whichITI=str(sys.argv[5])
trialsbacks=int(sys.argv[6])

def main():

	stimulus_set=np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
	num_stimpairs=np.shape(stimulus_set)[0]

	fig, axs = plt.subplots(1,trialsbacks-1,figsize=(1.25*trialsbacks,1.5))

	B11_all=[]
	B12_all=[]
	B21_all=[]
	B22_all=[]
	B_all=[]

	for t in range(1,trialsbacks):
		string = "%s_%s_ISI%s_ITI%s_trialsback%d"%(whichsubject, whichdistrib, whichdelay, whichITI, t)

		B11=np.loadtxt("data_processed/Bias11_%s.txt"%string)
		B12=np.loadtxt("data_processed/Bias12_%s.txt"%string)
		B21=np.loadtxt("data_processed/Bias21_%s.txt"%string)
		B22=np.loadtxt("data_processed/Bias22_%s.txt"%string)
		B=np.loadtxt("data_processed/Bias_%s.txt"%string)
		if t == 1:	
			B11_all = B11
			B12_all = B12
			B21_all = B21
			B22_all = B22
			B_all = B
		else:		
			B11_all=np.dstack((B11_all, B11))
			B12_all=np.dstack((B12_all, B12))
			B21_all=np.dstack((B21_all, B21))
			B22_all=np.dstack((B22_all, B22))
			B_all=np.dstack((B_all, B))

	# plot_B(B11_all, trialsbacks, 'B11', fig, axs, stimulus_set, color=cp.green)
	# plot_B(B12_all, trialsbacks, 'B12', fig, axs, stimulus_set, color=cp.blue)
	# plot_B(B21_all, trialsbacks, 'B21', fig, axs, stimulus_set, color=cp.red)
	# plot_B(B22_all, trialsbacks, 'B22', fig, axs, stimulus_set, color=cp.orange)
	plot_B(B_all, trialsbacks, 'B', fig, axs, stimulus_set, color='black')

	# fig.legend()
	fig.savefig("figs/bias_%s.png"%(string), bbox_inches='tight')
	fig.savefig("figs/bias_%s.svg"%(string), bbox_inches='tight')

	return

if __name__ == "__main__":
	main()





