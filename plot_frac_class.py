import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams
from matplotlib.ticker import FormatStrFormatter

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

def to_dB (x, x0=2.2e-7):
	return 10*np.log10(x/x0)

'''
min -> 60 = 10*log10(min/x0) --> x0 = min * 10^-6
'''
def plot_fc(stimulus_set, scattervals, performvals, num_stimpairs, dBsoundvals):
	fig, axs = plt.subplots(1,1,figsize=(1.5,1.25))
	# stimulus_set_dB = to_dB(stimulus_set)

	# scat=axs.scatter(stimulus_set_dB[:num_stimpairs,0], stimulus_set_dB[:num_stimpairs,1], marker='s', s=30, c=scattervals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)
	scat=axs.scatter(stimulus_set[:num_stimpairs,0], stimulus_set[:num_stimpairs,1], marker='s', s=30, c=scattervals, cmap=plt.cm.coolwarm, vmin=0, vmax=1)
	for i in range(int(num_stimpairs/2)):
		axs.text(stimulus_set[i,0]+0.1,stimulus_set[i,1]-0.3,'%d'%(performvals[i]*100))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		axs.text(stimulus_set[i,0]-0.4,stimulus_set[i,1]+0.15,'%d'%(performvals[i]*100))

	axs.plot(np.linspace(np.min(stimulus_set)-0.5,np.max(stimulus_set)+0.5,10),np.linspace(np.min(stimulus_set)-0.5,np.max(stimulus_set)+0.5,10), color='black', lw=0.5)
	axs.set_xlabel("Stimulus 1 (dB)")
	axs.set_ylabel("Stimulus 2 (dB)")
	B=np.unique(dBsoundvals)
	axs.set_xlim(np.min(stimulus_set)-0.5,np.max(stimulus_set)+0.5)
	axs.set_ylim(np.min(stimulus_set)-0.5,np.max(stimulus_set)+0.5)
	axs.set_xticks(np.unique(stimulus_set))
	axs.set_xticklabels([B[i] for i in np.arange(len(B))], rotation=45)
	axs.set_yticks(np.unique(stimulus_set))
	axs.set_yticklabels([B[i] for i in np.arange(len(B))], rotation=45)
	plt.colorbar(scat,ax=axs,ticks=[0,0.5,1])
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)

	return fig

whichsubject=sys.argv[1]
whichdistrib=sys.argv[2]
whichspecies=sys.argv[3]
whichdelay=str(sys.argv[4])
whichITI=str(sys.argv[5])

if __name__ == "__main__":

	stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))
	num_stimpairs=np.shape(stimulus_set)[0]
	dBsoundvals=np.loadtxt("data_processed/sounds_dB.txt")

	frac_classvals = np.loadtxt("data_processed/Frac_class_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI))
	performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI))

	# take the average over all subjects
	if whichsubject == 'AllSubjects':
		performvals=np.mean(performvals, axis=0)
		frac_classvals=np.mean(frac_classvals, axis=0)
	# print(frac_classvals)
	print(np.mean(performvals))

	fig=plot_fc(stimulus_set, frac_classvals, performvals, num_stimpairs, dBsoundvals)

	fig.savefig("figs/cbias_%s_%s_ISI%s_ITI%s.png"%(whichsubject, whichdistrib, whichdelay, whichITI), bbox_inches='tight')
	fig.savefig("figs/cbias_%s_%s_ISI%s_ITI%s.svg"%(whichsubject, whichdistrib, whichdelay, whichITI), bbox_inches='tight')
