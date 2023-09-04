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
whichdistrib=sys.argv[2]
whichspecies=sys.argv[3]
whichISI=sys.argv[4]

stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))
num_stimpairs=np.shape(stimulus_set)[0]

indices_biasplus=[0,1,8,9]
indices_biasminus=[2,3,4,5,6,7]

pos=[]
neg=[]

ITIvals=['low', 'high']
for ITI in ITIvals:
	print(ITI)
	performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichISI, ITI))
	performvals = np.mean(performvals, axis=0)
	print(np.shape(performvals))
	pos+=[np.nanmean(performvals[indices_biasplus])-np.nanmean(performvals[:num_stimpairs])]
	neg+=[np.nanmean(performvals[indices_biasminus])-np.nanmean(performvals[:num_stimpairs])]

print(ITIvals)
print(pos)
print(neg)

fig, axs = plt.subplots(1,1,figsize=(1,1))

axs.bar([0,2],[pos[i]*100 for i in range(len(pos))], color=cp.green)
axs.bar([0,2],[neg[i]*100 for i in range(len(neg))], color=cp.orange)

# axs.set_ylim(-15,15)
axs.set_xticks([0,2])
axs.set_xticklabels(['low', 'high'])
axs.axhline(0,color='k')
axs.set_xlabel("Inter-trial interval [s]")
axs.set_ylabel("$\%$ correct minus average")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
fig.savefig('figs/cbias_fITI_%s_%s.png'%(whichsubject, whichdistrib), bbox_inches='tight')
fig.savefig('figs/cbias_fITI_%s_%s.svg'%(whichsubject, whichdistrib), bbox_inches='tight')


