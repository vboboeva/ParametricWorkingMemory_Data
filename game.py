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
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import minimize
from scipy import stats

from data import network_stimulus_set, network_performvals, rats_stimulus_set, rats_performvals, ha_stimulus_set, ha_performvals, ht_stimulus_set, ht_performvals


# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}, size=14)
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
		'''
		How about taking the buffer at trial t to be the set of stimuli presented up to trial t - 1?
		'''

		p_buffer = np.random.rand(num_trials)

		readout = np.empty(num_trials, dtype=int)

		for trial in range(num_trials):

			sa, sb = stimuli[trial].copy()

			if p_buffer[trial] < eps:
				sa = buffer[trial, np.random.choice(2)]
			
			readout[trial] = get_label(sa, sb)

		return scatter(self.stimulus_set,stimuli,readout,labels)

	def simulate_extra (self, eps, eps1, num_trials=1000):

		assert isinstance(self.pi_extra, stats._distn_infrastructure.rv_frozen), \
			   "optional argument pi_extra must be continuous random variable"

		probas = self.weights/np.sum(self.weights)
		stimuli = self.stimulus_set[np.random.choice(self.N, p=probas, size=num_trials)]
		labels = np.array([get_label(sa, sb) for sa, sb in stimuli])
		buffer = self.stimulus_set[np.random.choice(len(self.stimulus_set), p=probas, size=num_trials)]
		'''
		How about taking the buffer at trial t to be the set of stimuli presented up to trial t - 1?
		'''

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


def plot_scatter(stimulus_set, scattervals, performvals, figurename, num_stimpairs):
	fig, axs = plt.subplots(1,1,figsize=(4,3.5), num=1, clear=True)

	scat=axs.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=100, c=scattervals[:num_stimpairs], cmap=plt.cm.coolwarm_r)

	for i in range(int(num_stimpairs/2)):
		axs.text(stimulus_set[i,0]-0.05,stimulus_set[i,1]-0.1,'%d'%(performvals[i]*100))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		axs.text(stimulus_set[i,0]-0.05,stimulus_set[i,1]+0.1,'%d'%(performvals[i]*100))

	axs.plot(np.linspace(0,1,10),np.linspace(0,1,10), color='black')
	axs.set_xlabel("$s_a$")
	axs.set_ylabel("$s_b$")

	plt.colorbar(scat,ax=axs)
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	fig.savefig("game/%s.png"%(figurename), bbox_inches='tight')

def plot_fit(xd,yd,xf,yf,figurename):
	fig, axs = plt.subplots(1,1,figsize=(3,3), num=1, clear=True)
	
	axs.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='blue', marker='.')
	axs.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='red', marker='.')	

	axs.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='blue')
	axs.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='red')
	
	axs.set_xlabel("$s_a$")
	axs.set_ylabel("performance")

	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	fig.savefig("game/%s.png"%(figurename), bbox_inches='tight')


def main():
	########################################################################### BEGIN PARAMETERS ############################################################################

	SimulationName='game'

	num_trials=100000 # number of trials within each session	
	# # define stimulus set
	# stimulus_set = np.array([ [0.3,0.2], [0.4,0.3], [0.5,0.4], [0.6,0.5], [0.7,0.6], [0.8,0.7], \
	# 							[0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8],])# \
	# 							#[0.45, 0.5], [0.55, 0.5], [0.5, 0.45], [0.5, 0.55] ])
	# # define probabilities for equally spaced stimuli and also psychometric stimuli
	# probas=np.zeros(len(stimulus_set)) 
	# probas[:12]=0.9
	# probas[12:]=0.1
	# probas=probas/np.sum(probas)

	# # run the simulation
	p_b=float(sys.argv[1])
	# game = Game(stimulus_set, weights = probas, pi_extra=stats.norm(np.mean(stimulus_set), np.std(stimulus_set)))

	# # exit()

	# num_stimpairs=game.N

	# np.random.seed(1987) #int(params[index,2])) #time.time)	

	# # performvals, scattervals = game.simulate(p_b, num_trials=num_trials)
	# performvals_analytic = 1. - p_b * game.prob_error # perf(p_b)

	# # PLOT the simulation and the analytical
	# figurename='sim_analytic'
	# plot_fit(stimulus_set[:,0],performvals,stimulus_set[:,0],performvals_analytic,figurename)

	XDATA=[network_stimulus_set, rats_stimulus_set, ha_stimulus_set, ht_stimulus_set]
	YDATA=[network_performvals, rats_performvals, ha_performvals, ht_performvals]
	labels=['net', 'rats', 'ha', 'ht']
	epsvals=[0.36,0.4,0.65,0.7]
	# FIT the data
	for i, stimulus_set in enumerate(XDATA):
		print("------------ {} ------------".format(labels[i]))
		performvals=YDATA[i]
		try:
			# game = Game(stimulus_set)
			# game.check_distributions()
			# fitted_param=game.fit_eps(performvals) #epsvals[i]
			# print(fitted_param)
			# performvals_analytic=game.performances(fitted_param)


			mean, std = np.mean(stimulus_set), np.std(stimulus_set)
			game = Game(stimulus_set, pi_extra=stats.norm(mean+2.*std, .1*std))
			performvals_analytic, _ = game.simulate_extra(epsvals[i], 0.9, num_trials=num_trials)

			plot_fit(stimulus_set[:,0],performvals,stimulus_set[:,0],performvals_analytic,'%s'%labels[i])
		except:
			raise ValueError("Something wrong with \"{}\"".format(labels[i]))

	return



if __name__ == "__main__":
	main()
