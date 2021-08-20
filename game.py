#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb

@author: vb
"""
import numpy as np
import random
import os
import time
import cProfile
import pstats
import sys
from numba import jit

import matplotlib.pyplot as plt
from matplotlib import cm, use
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import minimize
from scipy import stats

from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals


# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

def get_label (sa, sb):
	if sa > sb:
		return 1
	elif sa < sb:
		return 0
	else:
		return np.random.choice(2) 


def fit_cumul(xvals,a,b):	
	return a*xvals+b		

def scatter(stimulus_set,stimuli,readout,labels):
	# SORT performance by pair of stimuli (for performance by stimulus type)
	performvals=np.zeros(len(stimulus_set))
	scattervals=np.zeros(len(stimulus_set))

	for m in range(len(stimulus_set)):
		
		l=stimulus_set[m]
		indices=np.where(np.all(stimuli==l, axis=1))[0]
		if (len(indices) != 0):
			performvals[m]=len(np.where(readout[indices] == labels[indices])[0])/len(labels[indices])
			scattervals[m]=np.nanmean(readout[indices])
	return performvals, scattervals


class Game (object):
	def __init__ (self, stimulus_set, weights=None, pi=None, pi_extra=None):
		# check that stimulus_set is an array of shape (N, 2), with arbitrary N
		assert len(stimulus_set.shape) == 2 and stimulus_set.shape[1] == 2, \
			   "invalid type/shape for stimulus_set"
		self.stimulus_set = stimulus_set
		self.stimuli_vals = np.unique(stimulus_set)		# values of individual stimuli
		
		self.N = len(self.stimulus_set)
		if weights is None:
			self.weights = np.ones(self.N)
		else:
			assert len(weights.shape)==1 and len(weights)==self.N, "wrong number of items for weights"
			self.weights = weights

		self.set_pi(pi)
		self.pi_extra = pi_extra

	def set_pi(self, pi):
		
		if pi is None:
			stimuli_prob = np.zeros_like(self.stimuli_vals).astype(float)	# probabilities of individual stimuli
			_weights = np.vstack(2*[self.weights]).T

			for i, s in enumerate(self.stimuli_vals):
				# look in stimulus_set where this is contained
				ids = np.where(self.stimulus_set == s)

				# sum to the probability vector the weights of those points
				stimuli_prob[i] += np.sum(_weights[ids])

			stimuli_prob /= np.sum(stimuli_prob)

			self.pi = stimuli_prob
		elif isinstance(pi, np.ndarray):
			print("array")
			self.pi = pi / np.sum(pi)

		self.cdf_lsr = np.array([np.sum(self.pi[:i]) for i in range(len(self.pi))])
		self.cdf_gtr = np.array([np.sum(self.pi[i+1:]) for i in range(len(self.pi))])

		self.pi_dict = dict([(s, p) for s,p in zip(self.stimuli_vals, self.pi)])
		self.cdf_lsr_dict = dict([(s, p) for s,p in zip(self.stimuli_vals, self.cdf_lsr)])
		self.cdf_gtr_dict = dict([(s, p) for s,p in zip(self.stimuli_vals, self.cdf_gtr)])

	def check_distributions(self):
		# CHECKS
		for key, val in self.pi_dict.items():
			print("pi({:.2f}) = {:.3f}".format(key, val))
		print("")
		for key, val in self.cdf_lsr_dict.items():
			print("P(s < {:.2f}) = {:.3f}".format(key, val))
		print("")
		for key, val in self.cdf_gtr_dict.items():
			print("P(s > {:.2f}) = {:.3f}".format(key, val))
		print("")


	@property
	def prob_error (self):
		'''
		Computes the probability that, by replacing s_a with a random value
		sampled from the distribution pi (optional parameter), one gets 
		a wrong classification. 
		'''
		if not hasattr(self, '_prob_error'):
			_prob_error = np.zeros(self.N) # to return: one probability for each stimulus pair
			pi = self.pi

			pi_dict = self.pi_dict
			cdf_dict = self.cdf_lsr_dict
			_cdf_dict = self.cdf_gtr_dict

			for i, (sa,sb) in enumerate(self.stimulus_set):
				# depending on whether sa > sb or viceversa,
				# calculate probabilities of error
				if sa > sb:
					p = 0.5 * pi_dict[sb] + cdf_dict[sb]
				elif sa < sb:
					p = 0.5 * pi_dict[sb] + _cdf_dict[sb]
				else:
					raise ValueError("Unexpected sa = sb stimulus pair")

				_prob_error[i] = p

			self._prob_error = _prob_error

		return self._prob_error

	def simulate (self, eps, num_trials=1000):
		probas = self.weights/np.sum(self.weights)
		stimuli = self.stimulus_set[np.random.choice(self.N, p=probas, size=num_trials)]
		labels = np.array([get_label(sa, sb) for sa, sb in stimuli])
		buffer = self.stimulus_set[np.random.choice(len(self.stimulus_set), p=probas, size=num_trials)]

		p_buffer = np.random.rand(num_trials)

		readout = np.empty(num_trials, dtype=int)

		for trial in range(num_trials):

			sa, sb = stimuli[trial].copy()

			if p_buffer[trial] < eps:
				sa = buffer[trial, np.random.choice(2)]
			
			readout[trial] = get_label(sa, sb)

		return scatter(self.stimulus_set,stimuli,readout,labels)

	def simulate_history (self, eps, num_trials=1000, figurename="history"):
		probas = self.weights/np.sum(self.weights)
		stimuli = self.stimulus_set[np.random.choice(self.N, p=probas, size=num_trials)]
		labels = np.array([get_label(sa, sb) for sa, sb in stimuli])

		p_buffer = np.random.rand(num_trials)

		readout = np.empty(num_trials, dtype=int)

		for trial in range(num_trials):

			sa, sb = stimuli[trial].copy()

			if p_buffer[trial] < eps and trial > 0:
				sa = stimuli[trial-1, 1]
				# sa = stimuli[trial-1, np.random.choice(2)]
			
			readout[trial] = get_label(sa, sb)

		np.save("history_stimuli.npy", stimuli)
		np.save("history_readout.npy", readout)

		plot_history(self.stimulus_set, stimuli, readout,figurename=figurename)

		return scatter(self.stimulus_set,stimuli,readout,labels)
		

	def simulate_extra (self, eps, eps1, num_trials=1000):

		assert isinstance(self.pi_extra, stats._distn_infrastructure.rv_frozen), \
			   "optional argument pi_extra must be continuous random variable"

		probas = self.weights/np.sum(self.weights)
		stimuli = self.stimulus_set[np.random.choice(self.N, p=probas, size=num_trials)]
		labels = np.array([get_label(sa, sb) for sa, sb in stimuli])
		buffer = self.stimulus_set[np.random.choice(len(self.stimulus_set), p=probas, size=num_trials)]

		p_buffer = np.random.rand(num_trials)
		p_extra = np.random.rand(num_trials)

		readout = np.empty(num_trials, dtype=int)

		for trial in range(num_trials):

			sa, sb = stimuli[trial].copy()

			if p_buffer[trial] < eps:
				if p_extra[trial] < eps1:
					sa = self.pi_extra.rvs()
				else:
					sa = buffer[trial, np.random.choice(2)]
			
			readout[trial] = get_label(sa, sb)

		return scatter(self.stimulus_set,stimuli,readout,labels)


	def performances(self, eps):
		return 1. - eps * self.prob_error

	def fit_eps (self, performvals):

		def distance (eps):
			delta = self.performances(eps) - performvals 
			delta /= performvals 
			# print(np.sum(delta*delta))
			return np.sum(delta*delta)

		opt = minimize(distance, np.array([0.5]))
		print("optimal error probability = %.3f"%(opt['x'],))
		return opt['x']

	def plot_stimulus_distr (self, filename='stimulus_distr.svg'):
		fig, ax = plt.subplots(figsize=(2,2))
		ax.set_xlabel("Stimulus")
		ax.set_ylabel("Probability")
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_ylim([0,0.2])
		ax.set_xlim([0.1,.9])
		color='grey'
		# ax.plot(self.stimuli_vals, self.pi, 'o', ms='12', color=color)
		ax.vlines(self.stimuli_vals, 0, self.pi, color=color, lw=4)
		ax.set_yticks([0,0.1,0.2])
		ax.set_yticklabels([0,0.1,0.2])
		fig.savefig(filename,bbox_inches='tight')

def history(stimulus_set, stimuli, readout, num_stimpairs):
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


def plot_history(stimulus_set, stimuli, readout, figurename="history"):
	
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


	fig.savefig(figurename+"_mat.svg", bbox_inches="tight")


	fig, axs = plt.subplots(1,1,figsize=(2,2))#, num=1, clear=True)
	axs.axhline(0, color='k')
	xdata=np.arange(int(num_stimpairs/2))
	from scipy.optimize import curve_fit
	# LINEAR FIT 
	def func(xvals,a,b):	
		return a*xvals+b

	for i in range(num_stimpairs):
		ydata=B[:,i]*100
		axs.scatter(xdata, ydata, alpha=0.5, s=5)
		popt, pcov = curve_fit(func, xdata, ydata)
		axs.plot(xdata, func(xdata, popt[0], popt[1]), alpha=0.3)

	bias=np.mean(B*100, axis=1)
	popt, pcov = curve_fit(func, xdata, bias)

	axs.scatter(xdata, bias, color='black', s=5)
	axs.plot(xdata, func(xdata, popt[0], popt[1]), color='black')

	axs.set_xticks(np.arange(0,6)) 
	axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(6) ],  rotation=30)
	axs.set_ylim(-20,20)
	axs.set_xlabel("Previous trial")
	axs.set_ylabel("Bias stimulus 1 $>$ stimulus 2 ($\%$)")
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	fig.savefig(figurename+"_bias.svg", bbox_inches='tight')


def plot_scatter(stimulus_set, scattervals, performvals, figurename, num_stimpairs):

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
	fig.savefig("%s"%(figurename), bbox_inches='tight')

def plot_fit(xd,yd,xf,yf,figurename,eps=None):
	fig, ax = plt.subplots(1,1,figsize=(2,2))#, num=1, clear=True)
	
	ax.set_ylim([0.4,1.])
	# ax.set_xlim([0.15,0.85])

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='crimson', marker='.', label="Stim 1 $>$ Stim 2")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='royalblue', marker='.', label="Stim 1 $<$ Stim 2")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='crimson')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='royalblue')

	if eps is not None:
		ax.plot([xd[0],xd[0]], [-1, 0], color='black', label="fit, $\epsilon = %.2f$"%(eps))

	h, l = ax.get_legend_handles_labels()
	ax.legend(h,l,loc='lower center')
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("%s.png"%(figurename), bbox_inches='tight')
	fig.savefig("%s.svg"%(figurename), bbox_inches='tight')


def plot_fit_and_distribution (xd,yd,xf,yf,pi,figurename,eps=None):
	fig, ax = plt.subplots(1,1,figsize=(2,2))#, num=1, clear=True)
	
	ax.set_ylim([0.4,1.])
	# ax.set_xlim([0.15,0.85])

	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='crimson', marker='.', label="$s_a > s_b$")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='royalblue', marker='.', label="$s_a < s_b$")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='crimson')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='royalblue')

	if eps is not None:
		ax.plot([xd[0],xd[0]], [-1, 0], color='black', label="fit, $\epsilon = %.3f$"%(eps))

	vals = np.unique(xd.ravel())

	median = percentile_discrete(0.5, vals, pi)

	ax2 = ax.twinx()
	ax2.set_ylim([0,0.5])
	color='green'
	ax2.yaxis.label.set_color(color)
	# ax2.spines['right'].set_color(color)
	ax2.tick_params(axis='y', colors=color)
	ax2.vlines(median, 0, 1, color='black', lw=1, ls='--')
	ax2.vlines(vals, 0, pi, color=color, lw=4)
	ax2.set_yticks([0,0.25,0.5])
	ax2.set_yticklabels([0,0.25,0.5])
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")
	ax2.set_ylabel("Probability, $\pi$")

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("%s"%(figurename), bbox_inches='tight')


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



def main():
	########################################################################### BEGIN PARAMETERS ############################################################################

	SimulationName='game'
	stimulus_set = network_stimulus_set

	num_trials=100000 # number of trials within each session	

	w_factor = 1

	# # run the simulation
	p_b=float(sys.argv[1])

	weights = np.ones(len(stimulus_set)).reshape((2,-1))
	weights[:,len(weights[0])//2:] = w_factor
	weights = np.ravel(weights)

	game = Game(stimulus_set, weights=weights)
	game.plot_stimulus_distr(filename='figs/stimulus_distr_%.2f.svg'%w_factor)
	num_stimpairs=game.N

	np.random.seed(1987) #int(params[index,2])) #time.time)	

	figurename='sim_history'
	performvals, scattervals = game.simulate_history(p_b, num_trials=num_trials, figurename="figs/history_%.2f_%.2f"%(w_factor, p_b))
	plot_scatter(stimulus_set, scattervals, performvals, "figs/performance_%.2f_%.2f.svg"%(w_factor,p_b), num_stimpairs)
	performvals_analytic = 1. - p_b * game.prob_error

	# PLOT the simulation and the analytical
	# plot_fit(stimulus_set[:,0],performvals,stimulus_set[:,0],performvals_analytic,"figs/"+figurename+".svg")
	plot_fit_and_distribution(stimulus_set[:,0],performvals,stimulus_set[:,0],performvals_analytic,game.pi,"figs/"+figurename+"_%.2f_%.2f.svg"%(w_factor, p_b))

	# XDATA=[network_stimulus_set]#, rats_stimulus_set, ha_stimulus_set, ht_stimulus_set]
	# YDATA=[network_performvals, rats_performvals, ha_performvals, ht_performvals]
	# labels=['net', 'rats', 'ha', 'ht']
	# epsvals=[0.36,0.4,0.65,0.7]
	# # FIT the data
	# for i, stimulus_set in enumerate(XDATA):
	# 	print("------------ {} ------------".format(labels[i]))
	# 	performvals=YDATA[i]
	# 	try:
	# 		game = Game(stimulus_set)
	# 		game.check_distributions()
	# 		fitted_param=game.fit_eps(performvals) #epsvals[i]
	# 		print(fitted_param)
	# 		performvals_analytic=game.performances(fitted_param)


	# 		# mean, std = np.mean(stimulus_set), np.std(stimulus_set)
	# 		# game = Game(stimulus_set, pi_extra=stats.norm(mean+2.*std, .1*std))
	# 		# performvals_analytic, _ = game.simulate_extra(epsvals[i], 0.9, num_trials=num_trials)

	# 		plot_fit(stimulus_set[:,0],performvals,stimulus_set[:,0],performvals_analytic,'%s'%labels[i],fitted_param)
	# 	except:
	# 		raise ValueError("Something wrong with \"{}\"".format(labels[i]))

	return



if __name__ == "__main__":
	main()
