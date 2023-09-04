import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
import pandas
from pandas import DataFrame
import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
import color_palette as cp

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

whichsubject=sys.argv[1]
whichspecies=sys.argv[2]
whichITI='all'


indices_biasplus=[0,1,8,9]
indices_biasminus=[2,3,4,5,6,7]


ISIvals=[2,4,6]
DistrTypes=['NegSkewed', 'Bimodal_l2']
colors=['gray', 'black']

fig, axs = plt.subplots(1,1,figsize=(1,1))
for j, DistrType in enumerate(DistrTypes):
	pos=[]
	neg=[]
	stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, DistrType)))
	num_stimpairs=np.shape(stimulus_set)[0]
	for ISI in ISIvals:
		print(ISI)
		performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, DistrType, ISI, whichITI))
		performvals = np.mean(performvals, axis=0)
		print(np.shape(performvals))
		pos+=[np.nanmean(performvals[indices_biasplus])-np.nanmean(performvals[:num_stimpairs])]
		neg+=[np.nanmean(performvals[indices_biasminus])-np.nanmean(performvals[:num_stimpairs])]
	axs.bar(ISIvals+0.8*j*np.ones(len(ISIvals)),[pos[i]*100 for i in range(len(pos))], color=colors[j], label='%s'%j)
	axs.bar(ISIvals+0.8*j*np.ones(len(ISIvals)),[neg[i]*100 for i in range(len(neg))], color=colors[j])
# axs.set_ylim(-15,15)
axs.set_xticks(ISIvals)
axs.set_xticklabels(ISIvals)
axs.axhline(0,color='k', linewidth=0.5)
axs.set_xlabel("Delay interval [s]")
axs.set_ylabel("$\%$ correct minus average")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.legend()
fig.savefig('figs/cbias_fISI_%s.png'%(whichsubject),bbox_inches='tight')
fig.savefig('figs/cbias_fISI_%s.svg'%(whichsubject),bbox_inches='tight')


