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


# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}, size=14)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath


# LINEAR FIT 
def func(xvals,a,b):	
	return a*xvals+b

def make_data():
	stimulus_set=np.load( SimulationName+"/stimulus_set.npy")
	dstim_set=np.load( SimulationName+"/dstim_set.npy")			

	stimuli=np.empty((0,2), float)
	drift=np.empty((0,2), float)
	labels=np.empty((0,1), float)
	VWM=np.empty((0,2*N), float)
	takevals=np.append(np.arange(0,N),np.arange(2*N,3*N))

	for sim in range(0,num_sims):
		myfile=SimulationName+'/VWM_sim%d.npy'%sim
		#print(myfile)
		if os.path.isfile(myfile):
			stimuli=np.append(stimuli, np.load( SimulationName+"/stimuli_sim%d.npy"%sim), axis=0)
			drift=np.append(drift, np.load( SimulationName+"/drift_sim%d.npy"%sim), axis=0)
			labels=np.append(labels, np.load( SimulationName+"/label_sim%d.npy"%sim))
			VWM=np.append(VWM, np.load( SimulationName+"/VWM_sim%d.npy"%sim)[:,takevals], axis=0)

	readout=np.empty(len(stimuli), float)

	for i in range(len(stimuli)):
		x1=np.argmax(VWM[i,:N])
		x2=np.argmax(VWM[i,N:2*N])

		if x1 > x2:
			readout[i]=1
		else:
			readout[i]=0
	
	return stimulus_set, stimuli, readout	

def history():
	trialtypevals=np.zeros((len(stimulus_set), len(stimulus_set)))
	responsevals=np.zeros((len(stimulus_set), len(stimulus_set)))

	# SORT performance by previous pair of stimuli
	for idx in range(len(stimuli)):
		for m in range(len(stimulus_set)):
			if ( stimuli[idx]==stimulus_set[m] ).all():
				for n in range(len(stimulus_set)):
					if ( stimuli[idx-1]==stimulus_set[n] ).all():
						trialtypevals[n,m] += 1
						responsevals[n,m] += readout[idx]

	A1=responsevals[0:int(num_stimpairs/2),:num_stimpairs]/trialtypevals[0:int(num_stimpairs/2),:num_stimpairs]
	B1=np.zeros((int(num_stimpairs/2),num_stimpairs))
	for i in range(num_stimpairs):
		B1[:,i] = (A1[:,i] - np.mean(A1[:,i]))

	A2=responsevals[int(num_stimpairs/2):num_stimpairs,:num_stimpairs]/trialtypevals[int(num_stimpairs/2):num_stimpairs,:num_stimpairs]
	B2=np.zeros((int(num_stimpairs/2),num_stimpairs))
	for i in range(num_stimpairs):
		B2[:,i] = (A2[:,i] - np.mean(A2[:,i]))

	B=np.hstack((B1,B2))
	H=np.divide(responsevals, trialtypevals, out=np.zeros_like(responsevals), where=trialtypevals!=0)
	return B, H


tauhH=0.5#float(sys.argv[1])
tauthetaH=5#float(sys.argv[2])
DH=0.0#float(sys.argv[3])
AHtoWM=0.47#float(sys.argv[4])
ISI=2

N=2000
num_sims=10
num_stimpairs=12

# SimulationName="AHtoWM%.2f_tauhH%.2f_J0WM%.2f_J0H%.2f_TISI2"%(AHtoWM,tauhH,J0WM,J0H)
SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_TISI%d"%(AHtoWM,tauhH,tauthetaH,DH,ISI)

fig, axs = plt.subplots(1,1,figsize=(4,3))

stimulus_set, stimuli, readout=make_data()
B, H =history()

im=axs.imshow(H[:num_stimpairs,:num_stimpairs], cmap=cm.Purples)
#im=ax.imshow(responsevals[:8,:8], cmap=cm.Purples)
axs.tick_params(axis='x', direction='out')
axs.tick_params(axis='y', direction='out')
plt.colorbar(im, ax=axs, shrink=1.)
axs.set_xticks(np.arange(num_stimpairs))
axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(num_stimpairs) ] ,  rotation=90)
axs.set_yticks(np.arange(num_stimpairs))
axs.set_yticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(num_stimpairs) ] )
axs.set_xlabel("current trial")
axs.set_ylabel("previous trial")


fig.savefig("figs/history_%s.png"%(SimulationName), bbox_inches='tight')