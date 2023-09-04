import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
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
from helper_history import history

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

# make a custom colormap: this one resembles most closely to Athena's
norm=plt.Normalize(0,1)
mycmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["indigo","white"])

whichsubject=str(sys.argv[1])
whichdistrib=str(sys.argv[2])
whichspecies=str(sys.argv[3])
whichdelay=str(sys.argv[4])
whichITI=str(sys.argv[5])
trialsback=int(sys.argv[6])

def plot_H(filename, H, dBsoundvals):
	fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))
	num_stimpairs=np.shape(H)[0]

	im=axs.imshow(H[:num_stimpairs,:num_stimpairs], cmap=mycmap, vmin=0, vmax=1)
	axs.tick_params(axis='x', direction='out')
	axs.tick_params(axis='y', direction='out')
	plt.colorbar(im, ax=axs, shrink=0.9, ticks=[0,0.5,1])

	axs.set_xticks(np.arange(num_stimpairs/2))
	axs.set_xticklabels(['%.1f, %.1f'%(dBsoundvals[j,0], dBsoundvals[j,1] ) for j in range(5) ],  rotation=45, ha='right')

	# axs.set_xticklabels(['0.3,0.2','','','','','0.8,0.7','0.2,0.3','','','','','0.7,0.8'],  rotation=45)
	#axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(num_stimpairs) ] ,  rotation=90)

	axs.set_yticks(np.arange(num_stimpairs/2))
	axs.set_yticklabels(['%.1f, %.1f'%(dBsoundvals[j,0], dBsoundvals[j,1] ) for j in range(5) ], ha='right')
	# axs.set_yticklabels(['0.3,0.2','','','','','0.8,0.7','0.2,0.3','','','','','0.7,0.8'])

	axs.set_xlabel("Current trial pair ($s_1, s_2$)")
	axs.set_ylabel("Previous trial pair ($s_1, s_2$)")
	fig.savefig("figs/%s.png"%(filename), bbox_inches='tight')
	fig.savefig("figs/%s.svg"%(filename), bbox_inches='tight')

def main():

	dBsoundvals=np.loadtxt("data_processed/sounds_dB.txt")
	filename='History_%s_%s_ISI%s_ITI%s_trialsback%d'%(whichsubject, whichdistrib, whichdelay, whichITI, trialsback)
	H=np.loadtxt("data_processed/%s.txt"%filename)
	plot_H(filename, H, dBsoundvals)

	return

if __name__ == "__main__":
	main()