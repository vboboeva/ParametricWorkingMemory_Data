import sys
import numpy as np
import matplotlib.pyplot as plt
from game_functions import Game, Game_Bayes
import color_palette as cp
from game_functions import percentile_discrete
from os.path import join

def mean_squared_error(act, pred):
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   return mean_diff

if __name__ == "__main__":

	whichsubject=str(sys.argv[1])
	whichdistrib=str(sys.argv[2])
	whichspecies=str(sys.argv[3])
	whichloss=str(sys.argv[4])
	whichdelay=str(sys.argv[5])
	whichITI=str(sys.argv[6])

	'''
	OBTAIN FITTED PARAMETERS: EPS DELTA GAMMA

	For optional argument `model`
	- `eps` (fit re-sampling rate; default)
	- `eps_delta` (including lapse)
	- `full` (including lapse and exponential reweighting of pi)
	'''
	if whichspecies == "Net": 
		stimulus_set = np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
		performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(whichspecies, whichdistrib))
		# create 'pi' from data -- from PPC bump location -- option only possible with network 
		sample = np.load("data_processed/locbumpPPC.npy")	
		counts, bins = np.histogram(sample, bins=20, density=True)
		values = (bins[1:] + bins[:-1])/2
		probas = counts/np.sum(counts)
		pi = np.array([values, probas])
		game = Game(stimulus_set, pi=pi, model="eps", loss=whichloss) 

	elif whichspecies == "Rat":
		stimulus_set = np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
		performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(whichspecies, whichdistrib))
		game = Game(stimulus_set, weights=None, pi=None, model="eps_delta", loss=whichloss)
		pi = game.get_pi()

	elif whichspecies == "Human":

		if whichdistrib == "Uniform": 
			stimulus_set = np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
			performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(whichspecies, whichdistrib))
			game = Game(stimulus_set, pi=None, model="eps_delta", loss=whichloss)
			pi = game.get_pi()

		elif whichdistrib == 'NegSkewed':
			stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))
			performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI))			
			lam = np.log(5.)/(len(stimulus_set)//2 - 1)
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0])
			game = Game(stimulus_set, weights=weights, model="eps", loss=whichloss)
			pi = game.get_pi()
			gameB = Game_Bayes(stimulus_set, pi=pi, loss=whichloss)

		elif whichdistrib == 'Bimodal_l1':
			stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))
			performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI))			
			lam = 1.
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0 + weights0[::-1]])
			game = Game(stimulus_set, weights=weights, model="eps", loss=whichloss)
			pi = game.get_pi()
			gameB = Game_Bayes(stimulus_set, pi=pi, sigma=None, loss=whichloss)

		elif whichdistrib == 'Bimodal_l2':
			stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))
			performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI))			
			lam = 2.
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0 + weights0[::-1]])
			game = Game(stimulus_set, weights=weights, model="eps", loss=whichloss)
			pi = game.get_pi()
			gameB = Game_Bayes(stimulus_set, pi=pi, sigma=None, loss=whichloss)

	x=stimulus_set[:,0]
	if whichsubject=='AllSubjects' and whichdistrib != 'Uniform':
		yd=np.mean(performvals, axis=0)
	else:
		yd=performvals

	np.savetxt("data_processed/Pi_%s_%s.txt"%(whichspecies, whichdistrib), pi, fmt='%.3f')

	print('fit stat model')
	fitted_param=game.fit(yd)
	performvals_fit=game.performances(*fitted_param)
	yf=performvals_fit
	# print('MSE=', mean_squared_error(yd,yf))
	np.savetxt("data_processed/Performance_StatFit_%s_%s_%s_%s.txt"%(whichsubject, whichspecies, whichdistrib, whichloss), [x, yf], fmt='%.3f')
	np.savetxt("data_processed/MSE_StatFit_%s_%s_%s_%s.txt"%(whichsubject, whichspecies, whichdistrib, whichloss), [mean_squared_error(yd,yf)])

	if whichspecies == 'Human' and whichdistrib != 'Uniform':
		print('fit Bayes')
		fitted_param=gameB.fit(yd)
		performvals_fit=gameB.performances_bayes(fitted_param[0])
		x=stimulus_set[:,0]
		yf=performvals_fit
		# print('MSE=', mean_squared_error(yd,yf))
		np.savetxt("data_processed/Performance_BayesFit_%s_%s_%s_%s.txt"%(whichsubject, whichspecies, whichdistrib, whichloss), [x, yf], fmt='%.3f')
		np.savetxt("data_processed/MSE_BayesFit_%s_%s_%s_%s.txt"%(whichsubject, whichspecies, whichdistrib, whichloss), [mean_squared_error(yd,yf)])
