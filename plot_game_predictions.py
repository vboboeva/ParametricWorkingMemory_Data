from game_functions import Game, Game_Bayes
from game_functions import percentile_discrete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, use
from helper_set_pi import set_pi
import sys


def plot_gamefit_and_distribution(xd,yd,xf,yf,pi,eps,whichdistrib):
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))#, num=1, clear=True)
	
	ax.set_ylim([0.,1.])
	# ax.set_xlim([0.15,0.85])

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='crimson', marker='.', label="$s_a > s_b$")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='royalblue', marker='.', label="$s_a < s_b$")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='crimson')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='royalblue')

	if eps is not None:
		ax.plot([xd[0],xd[0]], [-1, 0], color='black', label="fit, $\epsilon=%.2f$"%eps)
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

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
	ax2.set_ylabel("Probability $p_m$")
	
	fig.savefig("figs/performance_fs1_stimdistrib_%s_eps%.2f.png"%(whichdistrib, eps), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_stimdistrib_%s_eps%.2f.svg"%(whichdistrib, eps), bbox_inches='tight')
	plt.close(fig)

def plot_gamefit_bayesfit_and_distribution(x, ya, ys, yb, pi, eps, whichdistrib):
	
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))#, num=1, clear=True)
	
	ax.set_ylim([0.0,1.])
	# ax.set_xlim([0.15,0.85])

	# plot simulation in dots
	ax.scatter(x[:len(x)//2], ys[:len(x)//2], color='royalblue', marker='.', label="$s_a > s_b$")
	ax.scatter(x[len(x)//2:], ys[len(x)//2:], color='crimson', marker='.', label="$s_a < s_b$")	
	
	# plot analytical in solid lines
	ax.plot(x[:len(x)//2], ya[:len(x)//2], color='royalblue')
	ax.plot(x[len(x)//2:], ya[len(x)//2:], color='crimson')

	# plot bayesian in dashed lines
	ax.plot(x[:len(x)//2], yb[:len(x)//2], color='royalblue', ls='--')
	ax.plot(x[len(x)//2:], yb[len(x)//2:], color='crimson', ls='--')

	if eps is not None:
		ax.plot([x[0],x[0]], [-1, 0], color='black', label="fit, $\epsilon=%.2f$"%eps)

	vals = np.unique(x.ravel())
	print(np.shape(vals), np.shape(pi))

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
	fig.savefig("figs/performance_fs1_%s_eps%.2f.png"%(whichdistrib,eps), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_%s_eps%.2f.svg"%(whichdistrib,eps), bbox_inches='tight')

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

def plot_frac_class(stimulus_set, performvals, condition, whichdistrib, eps):

	num_stimpairs=np.shape(stimulus_set)[0]

	frac_classvals=np.empty(num_stimpairs)
	frac_classvals[:int(num_stimpairs/2)] = 1.-performvals[:int(num_stimpairs/2)]
	frac_classvals[int(num_stimpairs/2):] = performvals[int(num_stimpairs/2):]

	fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))
	scat=axs.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=30, c=frac_classvals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

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
	
	fig.savefig("figs/frac_class_%s_%s_%s.png"%(condition, whichdistrib, eps), bbox_inches='tight')
	fig.savefig("figs/frac_class_%s_%s_%s.svg"%(condition, whichdistrib, eps), bbox_inches='tight')

if __name__ == "__main__":

	whichdistrib=str(sys.argv[1])
	eps = float(sys.argv[2])

	data=np.loadtxt("data_processed/Performance_ModelPredictions_%s_eps%.3f.txt"%(whichdistrib, eps))	

	stimulus_set=data[:,:2]
	performvals_analytic=data[:,2]
	performvals_sim=data[:,3]
	performvals_bayes=data[:,4]	

	if whichdistrib == 'Uniform':
		gamma = 1.
		weights = np.ones(len(stimulus_set))
		pi = set_pi( stimulus_set, weights )

	elif whichdistrib == 'PosSkewed':	
		lam = -np.log(5.)/(len(stimulus_set)//2 - 1)
		weights = np.exp(lam * np.arange(len(stimulus_set)//2))
		weights = np.hstack(2*[weights])
		pi = set_pi( stimulus_set, weights )
		gamma=np.exp(lam)

	elif whichdistrib == 'NegSkewed':
		lam = np.log(5.)/(len(stimulus_set)//2 - 1)
		weights = np.exp(lam * np.arange(len(stimulus_set)//2))
		weights = np.hstack(2*[weights])
		pi = set_pi( stimulus_set, weights )
		gamma=np.exp(lam)

	elif whichdistrib == 'Bimodal_l1':
		lam = 1; weights = np.exp(lam * np.arange(len(stimulus_set)//2))
		weights = np.hstack(2*[weights + weights[::-1]])
		pi = set_pi( stimulus_set, weights )
		gamma=np.exp(lam)

	elif whichdistrib == 'Bimodal_l2':
		lam = 2; weights = np.exp(lam * np.arange(len(stimulus_set)//2))
		weights = np.hstack(2*[weights + weights[::-1]])
		pi = set_pi( stimulus_set, weights )
		gamma=np.exp(lam)
	
	else:
		raise ValueError (f'\"{whichdistrib}\" not recognized! Please try again.')

	# plot performance curves on the same plot
	plot_gamefit_bayesfit_and_distribution(stimulus_set[:,0], performvals_analytic, performvals_sim, performvals_bayes, pi[1,:], eps, whichdistrib)

	# plot fraction classified as s1<s2 different figs
	plot_frac_class(stimulus_set, performvals_sim, 'game', whichdistrib, eps)
	plot_frac_class(stimulus_set, performvals_bayes, 'bayes', whichdistrib, eps)
