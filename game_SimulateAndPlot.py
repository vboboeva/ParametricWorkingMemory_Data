from game import Game, scatter, history #, plot_scatter, plot_fit, plot_fit_and_distribution
from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, use

def plot_scatter(stimulus_set, scattervals, performvals, eps, num_stimpairs):

	fig, axs = plt.subplots(1,1,figsize=(2.25,2))
	scat=axs.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=40, c=scattervals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

	for i in range(int(num_stimpairs/2)):
		axs.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.1,'%d'%(performvals[i]*100))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		axs.text(stimulus_set[i,0]-0.1,stimulus_set[i,1]+0.1,'%d'%(performvals[i]*100))

	axs.plot(np.linspace(0,1,10),np.linspace(0,1,10), color='black')
	axs.set_xlabel("Stimulus 1")
	axs.set_ylabel("Stimulus 2")
	axs.set_yticks([0,0.5,1])
	axs.set_yticklabels([0,0.5,1])
	plt.colorbar(scat,ax=axs,ticks=[0,0.5,1])
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	
	fig.savefig("figs/scatter_s1_s2_eps%.2f.png"%(eps), bbox_inches='tight')
	fig.savefig("figs/scatter_s1_s2_eps%.2f.svg"%(eps), bbox_inches='tight')

def plot_fit(xd,yd,xf,yf,eps):
	fig, ax = plt.subplots(1,1,figsize=(2,2))#, num=1, clear=True)
	
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

def plot_hmatrix_bias(stimulus_set, stimuli, readout, eps):
	
	num_stimpairs=len(stimulus_set)
	B, H =history(stimulus_set, stimuli, readout, num_stimpairs)

	fig, axs = plt.subplots(1,1,figsize=(2.2,2))

	im=axs.imshow(H[:num_stimpairs,:num_stimpairs], cmap=cm.Purples)
	#im=ax.imshow(responsevals[:8,:8], cmap=cm.Purples)
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


	fig, axs = plt.subplots(1,1,figsize=(2,2))#, num=1, clear=True)
	#axs.axhline(0, color='k')
	xdata=np.arange(int(num_stimpairs/2))
	from scipy.optimize import curve_fit
	# LINEAR FIT 
	def func(xvals,a,b):	
		return a*xvals+b

	for i in range(num_stimpairs):
		ydata=B[:,i]*100
		#axs.scatter(xdata, ydata, alpha=0.5, s=5)
		popt, pcov = curve_fit(func, xdata, ydata)
		axs.plot(xdata, func(xdata, popt[0], popt[1]), alpha=0.3)

	bias=np.mean(B*100, axis=1)
	popt, pcov = curve_fit(func, xdata, bias)

	axs.scatter(xdata, bias, color='black', s=5)
	axs.plot(xdata, func(xdata, popt[0], popt[1]), color='black')

	axs.set_xticks(np.arange(0,6)) 
	axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(6) ],  rotation=30)
	#axs.set_ylim(-25,25)
	axs.set_xlabel("Previous trial stim. pair $(s_1, s_2)$")
	axs.set_ylabel("Bias stimulus 1 $>$ stimulus 2 ($\%$)")
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	fig.savefig("figs/bias_eps%.2f.png"%(eps), bbox_inches="tight")
	fig.savefig("figs/bias_eps%.2f.svg"%(eps), bbox_inches="tight")

if __name__ == "__main__":

	SimulationName='game'
	stimulus_set = network_stimulus_set

	num_trials=100000 # number of trials within each session	
	num_stimpairs=12

	# run the simulation with these parameters, PRODUCE FIG 4F
	eps = 0.253 #float(sys.argv[1])

	game = Game(stimulus_set)
	np.random.seed(1987) #int(params[index,2])) #time.time)	

	# Simulate and plot scatter
	stimuli, readout, labels = game.simulate(eps, num_trials)
	performvals, scattervals = scatter(stimulus_set, stimuli, readout, labels) 
	plot_scatter(stimulus_set, scattervals, performvals, eps, num_stimpairs)

	# Simulate and plot history
	stimuli, readout, labels = game.simulate_history(eps, num_trials)
	plot_hmatrix_bias(stimulus_set, stimuli, readout, eps)

	# PLOT the simulation and the analytical with the error rate entered above
	performvals_analytic = 1. - eps * game.prob_error
	plot_fit(stimulus_set[:,0], performvals, stimulus_set[:,0], performvals_analytic, eps)
