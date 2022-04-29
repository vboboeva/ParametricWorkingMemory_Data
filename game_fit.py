from game import Game
from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals
import numpy as np
import matplotlib.pyplot as plt

def plot_fit(xd,yd,xf,yf,datatype,eps,delta,gamma):
	fig, ax = plt.subplots(1,1,figsize=(2,2))
	

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='crimson', marker='.')#, label="Stim 1 $>$ Stim 2")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='royalblue', marker='.')#, label="Stim 1 $<$ Stim 2")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='crimson')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='royalblue')

	# to put a legend
	if eps is not None:
		ax.plot([xd[0],xd[0]], [-1, 0], color='black', label="$\epsilon = %.2f$ \n $\delta=%.2f$ \n $\gamma=%.2f$"%(eps, delta, gamma))

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

