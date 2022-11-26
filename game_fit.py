from game_functions import Game
from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals
import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(stimulus_set, scattervals, performvals, num_stimpairs, whichspecies, datatype):

	fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))
	scat=axs.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=30, c=scattervals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

	for i in range(int(num_stimpairs/2)):
		axs.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.15,'%d'%(performvals[i]))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		axs.text(stimulus_set[i,0]-0.20,stimulus_set[i,1]+0.05,'%d'%(performvals[i]))

	axs.plot(np.linspace(0,1,10),np.linspace(0,1,10), color='black', lw=0.5)
	axs.set_xlabel("Stimulus 1")
	axs.set_ylabel("Stimulus 2")

	if whichspecies == 'net':
		axs.set_yticks([0,0.5,1])
		axs.set_yticklabels([0,0.5,1])
	elif whichspecies == 'rats':
		# axs.set_yticks([stimulus_set[0,:]])
		# axs.set_yticklabels([stimulus_set[0,:]])
		axs.set_xlim(50,100)
		axs.set_ylim(50,100)

	plt.colorbar(scat,ax=axs,ticks=[0,0.5,1])
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	
	fig.savefig("figs/scatter_s1_s2_%s_%s.png"%(whichspecies, datatype), bbox_inches='tight')
	fig.savefig("figs/scatter_s1_s2_%s_%s.svg"%(whichspecies, datatype), bbox_inches='tight')

def plot_fit(xd,yd,xf,yf,datatype,eps,delta=None,gamma=None):
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
	
	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='royalblue', marker='.')#, label="Stim 1 $>$ Stim 2")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='crimson', marker='.')#, label="Stim 1 $<$ Stim 2")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson')
	plt.axvline(np.mean(stimulus_set), ls='--', color='gray')

	# to put a legend
	label=""
	if eps is not None:
		label+="$\epsilon = %.2f$"%(eps,)
	if delta is not None:
		label+="\n$\delta=%.2f$"%(delta,)
	if gamma is not None:
		label+="\n$\gamma=%.2f$"%(gamma,)
	
	ax.plot([xd[0],xd[0]], [-1, 0], color='black', label=label)

	h, l = ax.get_legend_handles_labels()
	ax.legend(h, l, loc='best')
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")

	ax.set_ylim([0.4,1.])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("figs/performance_fs1_%s.png"%(datatype), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_%s.svg"%(datatype), bbox_inches='tight')



if __name__ == "__main__":

	SimulationName='game'
	stimulus_set = network_stimulus_set

	num_trials=100000 # number of trials within each session	

	game = Game(stimulus_set)
	num_stimpairs=game.N

	np.random.seed(1987) #int(params[index,2])) #time.time)	

	# OBTAIN FITTED PARAMETERS: EPS DELTA GAMMA

	XDATA=[rats_stimulus_set, ha_stimulus_set, ht_stimulus_set, network_stimulus_set] 
	YDATA=[rats_performvals, ha_performvals, ht_performvals, network_performvals]
	labels=['rats', 'ha', 'ht', 'net']

	# FIT the data: PRODUCE FIG 4D LEFT AND RIGHT
	for i, stimulus_set in enumerate(XDATA):
		print("------------ {} ------------".format(labels[i]))
		performvals=YDATA[i]
		try:
			game = Game(stimulus_set)
			fitted_param=game.fit(performvals)
			print(fitted_param)
			performvals_analytic=game.performances(*fitted_param)
			plot_fit(stimulus_set[:,0],performvals,stimulus_set[:,0],performvals_analytic,'%s'%labels[i],fitted_param[0],fitted_param[1],fitted_param[2])
		except:
			raise ValueError("Something wrong with \"{}\"".format(labels[i]))

