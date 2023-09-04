from game_functions import Game
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, use
from matplotlib import rc
from pylab import rcParams
import sys
from helper_compute_performance import compute_performance
from helper_history import history
from scipy.optimize import curve_fit

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

def plot_frac_class(stimulus_set, frac_classvals, performvals, eps, num_stimpairs):

	fig, ax = plt.subplots(1,1,figsize=(1.75,1.5))
	scat=ax.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=40, c=frac_classvals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

	for i in range(int(num_stimpairs/2)):
		ax.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.1,'%d'%(performvals[i]*100))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		ax.text(stimulus_set[i,0]-0.1,stimulus_set[i,1]+0.1,'%d'%(performvals[i]*100))

	min_range = np.min(stimulus_set)-(np.max(stimulus_set)-np.min(stimulus_set))/5.
	max_range = np.max(stimulus_set)+(np.max(stimulus_set)-np.min(stimulus_set))/5.
	ax.plot(np.linspace(min_range, max_range,10),np.linspace(min_range, max_range,10), color='black', lw=0.5)

	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Stimulus 2")
	# ax.set_yticks([0,0.5,1])
	# ax.set_yticklabels([0,0.5,1])
	plt.colorbar(scat,ax=ax,ticks=[0,0.5,1])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	
	fig.savefig("figs/scatter_s1_s2_eps%.2f.png"%(eps), bbox_inches='tight')
	fig.savefig("figs/scatter_s1_s2_eps%.2f.svg"%(eps), bbox_inches='tight')

def plot_fit(xd,yd,xf,yf,eps):
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))#, num=1, clear=True)
	
	ax.set_ylim([0.4,1.])
	# ax.set_xlim([0.15,0.85])

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='royalblue', marker='.')#, label="Stim 1 $>$ Stim 2")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='crimson', marker='.')#, label="Stim 1 $<$ Stim 2")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson')

	h, l = ax.get_legend_handles_labels()
	ax.legend(h, l, loc='best')
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("figs/performance_fs1_eps%.2f.png"%(eps), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_eps%.2f.svg"%(eps), bbox_inches='tight')


def linfunc(xvals,a,b):	
	return a*xvals+b

def plot_hmatrix_bias(stimulus_set, stimuli, readout, labels, eps):
	
	num_stimpairs=len(stimulus_set)
	B11, B12, B21, B22, B, H = history(1, stimuli, readout, stimulus_set, num_stimpairs)

	fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))

	im=axs.imshow(H[:num_stimpairs,:num_stimpairs], cmap=mycmap, vmin=0, vmax=1)
	axs.tick_params(axis='x', direction='out')
	axs.tick_params(axis='y', direction='out')
	plt.colorbar(im, ax=axs, shrink=0.9, ticks=[0,0.5,1])

	axs.set_xticks(np.arange(num_stimpairs))
	axs.set_xticklabels(['0.3,0.2','','','','','','0.2,0.3','','','','',''] ,  rotation=45)
	#axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(num_stimpairs) ] ,  rotation=90)

	axs.set_yticks(np.arange(num_stimpairs))
	axs.set_yticklabels(['0.3,0.2','','','','','','0.2,0.3','','','','',''] )
	axs.set_xlabel("Current trial")
	axs.set_ylabel("Previous trial")

	fig.savefig("figs/matrix_eps%.2f.png"%(eps), bbox_inches="tight")
	fig.savefig("figs/matrix_eps%.2f.svg"%(eps), bbox_inches="tight")


	fig, axs = plt.subplots(1,1,figsize=(1.5,1.5))#, num=1, clear=True)
	#axs.axhline(0, color='k')
	xdata=np.arange(int(num_stimpairs/2))

	for i in range(int(num_stimpairs/2)):
		ydata=B11[:,i]*100
		#axs.scatter(xdata, ydata, alpha=0.5, s=5)
		popt, pcov = curve_fit(linfunc, xdata, ydata)
		axs.plot(xdata, linfunc(xdata, popt[0], popt[1]), alpha=0.3)

	bias=np.mean(B*100, axis=1)
	popt, pcov = curve_fit(linfunc, xdata, bias)

	axs.scatter(xdata, bias, color='black', s=5)
	axs.plot(xdata, linfunc(xdata, popt[0], popt[1]), color='black')

	axs.set_xticks(np.arange(0,6)) 
	axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(6) ],  rotation=30)
	#axs.set_ylim(-25,25)
	axs.set_xlabel("Previous trial stim. pair $(s_1, s_2)$")
	axs.set_ylabel("Bias stimulus 1 $>$ stimulus 2 ($\%$)")
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	fig.savefig("figs/bias_eps%.2f.png"%(eps), bbox_inches="tight")
	fig.savefig("figs/bias_eps%.2f.svg"%(eps), bbox_inches="tight")


''' Simulates and plots results from the simple statistical model with free parameter eps'''

if __name__ == "__main__":

	whichdistrib='Uniform'
	whichspecies='Net'

	stimulus_set = np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))

	num_trials=100000 # number of trials within each session	
	num_stimpairs=np.shape(stimulus_set)[0]

	# run the simulation with these parameters
	eps = 0.253 #float(sys.argv[1])

	game = Game(stimulus_set)
	np.random.seed(1987) #int(params[index,2])) #time.time)	

	# Simulate and plot fraction classified for each stim pair
	stimuli, readout, labels = game.simulate(eps, num_trials)
	performvals = compute_performance(stimulus_set, stimuli, readout, labels)
	frac_classvals=np.empty(num_stimpairs)
	frac_classvals[:int(num_stimpairs/2)] = 1.-performvals[:int(num_stimpairs/2)]
	frac_classvals[int(num_stimpairs/2):] = performvals[int(num_stimpairs/2):]	 
	plot_frac_class(stimulus_set, frac_classvals, performvals, eps, num_stimpairs)

	# PLOT the simulation and the analytical with the error rate entered above
	performvals_analytic = 1. - eps * game.prob_error
	plot_fit(stimulus_set[:,0], performvals, stimulus_set[:,0], performvals_analytic, eps)

	# Simulate and plot history
	stimuli, readout, labels = game.simulate_history(eps, num_trials)
	plot_hmatrix_bias(stimulus_set, stimuli, readout, labels, eps)

