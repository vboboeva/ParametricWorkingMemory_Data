from game_functions import Game, Game_Bayes
from helper_compute_performance import compute_performance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, use
from helper_percentile_discrete import percentile_discrete
from helper_set_pi import set_pi
import sys

if __name__ == "__main__":

	whichdistrib=str(sys.argv[1])
	eps = float(sys.argv[2]) 
	# number of trials within each session
	num_trials=100000

	stimulus_set = 	np.loadtxt("data_processed/StimSet_Net_Uniform.txt")
	
	# run the simulation with these parameters

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

	# Get analytic values
	game = Game(stimulus_set, weights=np.ravel(weights))
	num_stimpairs=game.N

	# Analytical with the error rate entered above
	performvals_analytic = 1. - eps * game.prob_error

	# Simulate game
	np.random.seed(1987) 
	stimuli, readout, labels = game.simulate(eps, num_trials) 
	performvals_sim = compute_performance(stimulus_set, stimuli, readout, labels)

	# Fit stat model with the Bayesian model, find best_sigma
	gb = Game_Bayes(stimulus_set, pi=pi)
	best_sigma, = gb.fit(performvals_sim)
	performvals_bayes = gb.performances_bayes(best_sigma)

	print(gamma, best_sigma)
	
	# Save all the data to file
	datatosave = np.column_stack((stimulus_set, performvals_analytic, performvals_sim, performvals_bayes))
	np.savetxt("data_processed/Performance_ModelPredictions_%s_eps%.3f.txt"%(whichdistrib, eps), datatosave, fmt='%.3f')	

