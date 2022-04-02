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

#from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals


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
		self.gamma = 1
		self.delta = 0

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

			self.set_dicts(self.pi)

	def set_dicts (self, pi):
		self.cdf_lsr = np.array([np.sum(pi[:i]) for i in range(len(pi))])
		self.cdf_gtr = np.array([np.sum(pi[i+1:]) for i in range(len(pi))])

		self.pi_dict = dict([(s, p) for s,p in zip(self.stimuli_vals, pi)])
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
		# if not hasattr(self, '_prob_error'):
		_prob_error = np.zeros(self.N) # to return: one probability for each stimulus pair
		pi = self.pi.copy()

		# use gamma as exponential (geometric) weight to reweight pi
		if self.gamma != 1:
			reweighting = np.ones_like(self.pi)
			for i in range(1,len(reweighting)):
				reweighting[i] = reweighting[i-1]*self.gamma
			pi = pi*reweighting
			pi /= np.sum(pi)

		self.set_dicts(pi)

		pi_dict = self.pi_dict
		cdf_dict = self.cdf_lsr_dict
		_cdf_dict = self.cdf_gtr_dict

		for i, (sa, sb) in enumerate(self.stimulus_set):
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

		return stimuli, readout, labels

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

		return stimuli, readout, labels

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


	def performances(self, eps, delta, gamma):
		self.gamma = gamma
		# self.eps = eps
		# self.delta = delta
		return 1. - eps * self.prob_error - delta

	def fit (self, performvals):

		def distance (pars):
			eps, delta, gamma = pars
			relative_error = self.performances(eps, delta, gamma) - performvals 
			relative_error /= performvals 
			return np.sum(relative_error*relative_error)

		opt = minimize(distance, np.array([0.5,0,1.]))
		print("optimal error probability eps = %.3f"%(opt['x'][0],))
		print("optimal lapse parameter delta = %.3f"%(opt['x'][1],))
		print("optimal exponential factor gamma = %.3f"%(opt['x'][2],))
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





