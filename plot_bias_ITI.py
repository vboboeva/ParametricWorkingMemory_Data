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
import sys
import color_palette as cp

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

whichsubject=str(sys.argv[1])
whichdistrib=str(sys.argv[2])
whichspecies=str(sys.argv[3])
whichdelay=str(sys.argv[4])
trialsbacks=int(sys.argv[5])

def linfunc(xvals,a,b):	
	return a*xvals+b

def plot_B(B, trialsbacks, filename, fig, axs, dBsoundvals, num_stimpairs, color, label):
	
	for i, o in enumerate(range(1, trialsbacks)):

		# plot the mean
		xdata=np.arange(int(num_stimpairs/2))
		bias=np.mean(B[:,:,i]*100, axis=1)
		popt, pcov = curve_fit(linfunc, xdata, bias)
		axs[i].scatter(xdata, bias, color=color, s=5, label=label)
		axs[i].plot(xdata, linfunc(xdata, popt[0], popt[1]), color=color, label=label)

		# # plot individual curves
		# for j in range(int(num_stimpairs/2)):
		# 	ydata=B[:,j,i]*100
		# 	# axs[i].scatter(xdata, ydata, alpha=0.5, s=5)
		# 	popt, pcov = curve_fit(linfunc, xdata, ydata)
		# 	axs[i].plot(xdata, linfunc(xdata, popt[0], popt[1]), alpha=0.3)

		axs[i].set_xticks(np.arange(0,5)) 
		axs[i].set_xticklabels(['%.1f $\leftrightarrow$ %.1f'%(dBsoundvals[j,0], dBsoundvals[j,1] ) for j in range(5) ],  rotation=45, ha='right')
		axs[i].spines['right'].set_visible(False)
		axs[i].spines['top'].set_visible(False)
		axs[i].set_ylim(-15,15)

		if i==0:
			axs[i].set_ylabel("Stim 1 $<$ Stim 2 bias ($\%$)")
			axs[i].set_xlabel("%d trial back pair $(s_1, s_2)$"%o)
		else:
			axs[i].set_xticklabels([])
			axs[i].set_yticklabels([])
			axs[i].set_xlabel("%d trials back"%o)
	axs[i].legend()

def main():
	stimulus_set=np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
	dBsoundvals=np.loadtxt("data_processed/sounds_dB.txt")

	num_stimpairs=np.shape(stimulus_set)[0]
	ITIvals=['low',  'high']#, 'all']
	labelvals=['Low ITI',  'High ITI']#, 'All']
	colorvals=[cp.violet1,  cp.violet3]#, 'black']
	
	# ITIvals=['low', 'all']
	# labelvals=['Low ITI', 'All']
	# colorvals=[cp.violet1, 'black']


	fig, axs = plt.subplots(1,trialsbacks-1,figsize=(trialsbacks,1.25))

	# plot separarely for each ISI
	for i, ITI in enumerate(ITIvals):
		print(ITI)
		color=colorvals[i]
		B11_all=[]
		B12_all=[]
		B21_all=[]
		B22_all=[]
		B_all=[]
		
		for t in range(1,trialsbacks):
			string = "%s_%s_ISI%s_ITI%s_trialsback%d"%(whichsubject, whichdistrib, whichdelay, ITI, t)
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

		# plot_B(B11_all, trialsbacks, 'B11', fig, axs, color=cp.green)
		# plot_B(B12_all, trialsbacks, 'B12', fig, axs, color=cp.blue)
		# plot_B(B21_all, trialsbacks, 'B21', fig, axs, color=cp.red)
		# plot_B(B22_all, trialsbacks, 'B22', fig, axs, color=cp.orange)
		plot_B(B_all, trialsbacks, 'B', fig, axs, dBsoundvals, num_stimpairs, color=colorvals[i], label=labelvals[i])

	# plt.legend()
	fig.savefig("figs/bias_%s_fITI.png"%(string), bbox_inches='tight')
	fig.savefig("figs/bias_%s_fITI.svg"%(string), bbox_inches='tight')

	return

if __name__ == "__main__":
	main()

