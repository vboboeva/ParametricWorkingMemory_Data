from game_functions import Game, scatter, Game_Bayes
from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, use

def percentile_discrete(f, vals, probs):
	assert vals.shape == probs.shape, "invalid shapes of values and probabilities"
	c = 0
	v = None
	for i, (x,p) in enumerate(zip(vals, probs)):
		v = x
		c += p
		if c >= f:
			break
	return v

def plot_scatter(stimulus_set, scattervals, performvals, eps, gamma, num_stimpairs,condition):

	fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))
	scat=axs.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=30, c=scattervals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

	for i in range(int(num_stimpairs/2)):
		axs.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.15,'%d'%(performvals[i]*100))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		axs.text(stimulus_set[i,0]-0.20,stimulus_set[i,1]+0.05,'%d'%(performvals[i]*100))

	axs.plot(np.linspace(0,1,10),np.linspace(0,1,10), color='black', lw=0.5)
	axs.set_xlabel("Stimulus 1")
	axs.set_ylabel("Stimulus 2")
	axs.set_yticks([0,0.5,1])
	axs.set_yticklabels([0,0.5,1])
	plt.colorbar(scat,ax=axs,ticks=[0,0.5,1])
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	
	fig.savefig("figs/scatter_s1_s2_eps%.2f_gamma%.2f_%s.png"%(eps,gamma,condition), bbox_inches='tight')
	fig.savefig("figs/scatter_s1_s2_eps%.2f_gamma%.2f_%s.svg"%(eps,gamma,condition), bbox_inches='tight')

def plot_gamefit_and_distribution(xd,yd,xf,yf,pi,eps,gamma):
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))#, num=1, clear=True)
	
	ax.set_ylim([0.4,1.])
	# ax.set_xlim([0.15,0.85])

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='crimson', marker='.', label="$s_a > s_b$")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='royalblue', marker='.', label="$s_a < s_b$")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='crimson')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='royalblue')

	if eps is not None:
		ax.plot([xd[0],xd[0]], [-1, 0], color='black', label="fit, $\epsilon=%.2f$"%eps)

	vals = np.unique(xd.ravel())

	median = percentile_discrete(0.5, vals, pi)

	ax2 = ax.twinx()
	ax2.set_ylim([0,0.5])
	color='gray'
	ax2.yaxis.label.set_color(color)
	# ax2.spines['right'].set_color(color)
	ax2.tick_params(axis='y', colors=color)
	ax2.vlines(median, 0, 1, color='black', lw=1, ls='--')
	ax2.vlines(vals, 0, pi, color=color, lw=4)
	ax2.set_yticks([0,0.25,0.5])
	ax2.set_yticklabels([0,0.25,0.5])
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")
	ax2.set_ylabel("Probability $p_m$")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("figs/performance_fs1_stimdistrib_eps%.2f_gamma%.2f.png"%(eps,gamma), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_stimdistrib_eos%.2f_gamma%.2f.svg"%(eps,gamma), bbox_inches='tight')

def plot_gamefit_bayesfit_and_distribution(xd,yd,xf,yf,pi,eps,gamma,xb,yb):
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))#, num=1, clear=True)
	
	ax.set_ylim([0.4,1.])
	# ax.set_xlim([0.15,0.85])

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='royalblue', marker='.', label="$s_a > s_b$")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='crimson', marker='.', label="$s_a < s_b$")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson')

	ax.plot(xb[:len(xb)//2], yb[:len(xb)//2], color='royalblue', ls='--')
	ax.plot(xb[len(xb)//2:], yb[len(xb)//2:], color='crimson', ls='--')

	if eps is not None:
		ax.plot([xd[0],xd[0]], [-1, 0], color='black', label="fit, $\epsilon=%.2f$"%eps)

	vals = np.unique(xd.ravel())

	median = percentile_discrete(0.5, vals, pi)

	ax2 = ax.twinx()
	ax2.set_ylim([0,0.5])
	color='gray'
	ax2.yaxis.label.set_color(color)
	# ax2.spines['right'].set_color(color)
	ax2.tick_params(axis='y', colors=color)
	ax2.vlines(median, 0, 1, color='black', lw=1, ls='--')
	ax2.vlines(vals, 0, pi, color=color, lw=4)
	ax2.set_yticks([0,0.25,0.5])
	ax2.set_yticklabels([0,0.25,0.5])
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")
	ax2.set_ylabel("Probability $p_m$")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("figs/performance_fs1_stimdistrib_eps%.2f_gamma%.2f.png"%(eps,gamma), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_stimdistrib_eps%.2f_gamma%.2f.svg"%(eps,gamma), bbox_inches='tight')

def plot_post_proba_bayes(s_vals, p_label):
	fig, ax = plt.subplots()
	ax.set_xlabel("First stimulus")
	ax.set_ylabel("Second stimulus")
	ax.set_yticks(np.arange(7))
	ax.set_xticks(np.arange(7))
	ax.set_yticklabels(s_vals.astype(str))
	ax.set_xticklabels(s_vals.astype(str))
	im = ax.imshow(p_label.T, origin='lower', cmap="RdBu")
	ax.plot([-.5,6.5],[-.5,6.5], ls='--', c='k')
	fig.colorbar(im)
	plt.show()

if __name__ == "__main__":

	SimulationName='game'
	stimulus_set = network_stimulus_set
	
	# number of trials within each session
	num_trials=100000 
	distribtype='sym'

	# run the simulation with these parameters
	eps = 0.25 
	weights = np.ones(len(stimulus_set)).reshape((2,-1))

	if distribtype == 'sym':
		gamma = 1.

	elif distribtype == 'pos_skewed':	
		gamma = 1./3.
		weights[:,len(weights[0])//2:] *= gamma

	elif distribtype == 'neg_skewed':
		gamma = 3.	
		weights[:,len(weights[0])//2:] *= gamma

	elif distribtype == 'bimodal':
		gamma = 1.
		weights *= 0.1 #1.e-6
		weights[:,0] = 1.
		weights[:,-1] = 1.
	else:
		raise ValueError (f'\"{distribtype}\" not recognized! Please try again.')

	weights = np.ravel(weights)

	# Get analytic values
	game = Game(stimulus_set, weights=weights)
	num_stimpairs=game.N

	np.random.seed(1987) 

	# Simulate game
	stimuli, readout, labels = game.simulate(eps, num_trials) 

	# Plot
	performvals, scattervals = scatter(stimulus_set, stimuli, readout, labels) 
	plot_scatter(stimulus_set, scattervals, performvals, eps, gamma, num_stimpairs, 'game')

	# PLOT the simulation and the analytical with the error rate entered above
	performvals_analytic = 1. - eps * game.prob_error
	# plot_gamefit_and_distribution(stimulus_set[:,0], performvals, stimulus_set[:,0], performvals_analytic, game.pi, eps, gamma)

	# Get values according to Bayes

	sigma=0.08
	gb = Game_Bayes(stimulus_set, weights=weights, sigma=sigma)

	p_label = gb.response_probability()

	performvals_bayes = np.zeros(len(stimulus_set))
	scattervals_bayes = np.zeros(len(stimulus_set))
	s_vals = list(gb.s_vals)
	
	for k, (s1,s2) in enumerate(stimulus_set):
		i = s_vals.index(s1)
		j = s_vals.index(s2)
		scattervals_bayes[k] = 1. - p_label[i,j]
		performvals_bayes[k] = p_label[i,j] if s1 > s2 else 1. - p_label[i,j]

	# plot everything
	plot_gamefit_bayesfit_and_distribution(stimulus_set[:,0], performvals, stimulus_set[:,0], performvals_analytic, game.pi, eps, gamma, stimulus_set[:,0], performvals_bayes)

	plot_scatter(stimulus_set, scattervals_bayes, performvals_bayes, sigma, gamma, num_stimpairs, 'bayes')
