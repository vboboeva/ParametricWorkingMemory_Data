import sys
import numpy as np
import matplotlib.pyplot as plt
from game_functions import Game
from game_functions import Game_Bayes
from helper_plot_scatter import plot_scatter
from helper_plot_fit import plot_fit
import color_palette as cp
from helper_percentile_discrete import percentile_discrete
from helper_set_pi import set_pi

def mean_squared_error(act, pred):
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   return mean_diff

if __name__ == "__main__":

	SubjectName=str(sys.argv[1])
	DistrType=str(sys.argv[2])
	species=str(sys.argv[3])
	loss=str(sys.argv[4])

	print(SubjectName)

	stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(species,DistrType)))
	performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(SubjectName,DistrType))

	# OBTAIN FITTED PARAMETERS: EPS DELTA GAMMA

	if species == "Net": 
		# create 'pi' from data -- from PPC bump location -- option only possible with network 
		sample = np.load("mymaxPPC.npy")	
		counts, bins = np.histogram(sample, bins=20, density=True)
		values = (bins[1:] + bins[:-1])/2
		probas = counts/np.sum(counts)
		pi = np.array([values, probas])
		game = Game(stimulus_set, pi=pi, model="eps", loss=loss) 
		# game = Game(stimulus_set, pi=pi, model="eps_delta")
		SubjectName=species

	elif species == "Rat": 
		# game = Game(stimulus_set, pi=None, model="eps")
		game = Game(stimulus_set, weights=None, pi=None, model="eps_delta", loss=loss)
		# game = Game(stimulus_set, pi=None, model="full")
		SubjectName=species

	elif species == "Human":
		if DistrType == "Uniform": 
			# game = Game(stimulus_set, pi=None, model="eps", loss=loss)
			game = Game(stimulus_set, pi=None, model="eps_delta", loss=loss)
			gameB = Game_Bayes(stimulus_set, pi=None, loss=loss)
			# game = Game(stimulus_set, pi=None, model="full", loss=loss)
		elif DistrType == 'NegSkewed':
			lam = np.log(5.)/(len(stimulus_set)//2 - 1)
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0])
			pi = set_pi( stimulus_set, weights )
			game = Game(stimulus_set, weights=weights, model="eps", loss=loss)
			# game = Game(stimulus_set, weights=weights, model="eps_delta", loss=loss)
			# game = Game(stimulus_set, pi=pi, model="full", loss=loss)
			gameB = Game_Bayes(stimulus_set, pi=pi, loss=loss)
		elif DistrType == 'Bimodal_l1':
			lam = 1.
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0 + weights0[::-1]])
			pi = set_pi( stimulus_set, weights )
			game = Game(stimulus_set, weights=weights, model="eps", loss=loss)
			# game = Game(stimulus_set, weights=weights, model="eps_delta", loss=loss)
			# game = Game(stimulus_set, pi=pi, model="full", loss=loss)
			gameB = Game_Bayes(stimulus_set, pi=pi, sigma=None, loss=loss)

		elif DistrType == 'Bimodal_l2':
			lam = 2.
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0 + weights0[::-1]])
			pi = set_pi( stimulus_set, weights )
			game = Game(stimulus_set, weights=weights, model="eps", loss=loss)
			# game = Game(stimulus_set, weights=weights, model="eps_delta", loss=loss)
			# game = Game(stimulus_set, pi=pi, model="full", loss=loss)
			gameB = Game_Bayes(stimulus_set, pi=pi, sigma=None, loss=loss)

	xd=stimulus_set[:,0]
	if SubjectName=='AllSubjects' and DistrType != 'Uniform':
		yd=np.mean(performvals, axis=0)
	else:
		yd=performvals

	np.savetxt("data_processed/Pi_%s.txt"%(DistrType), pi, fmt='%.3f')

	print('fit stat model')
	fitted_param=game.fit(yd)
	performvals_fit=game.performances(*fitted_param)
	xf=stimulus_set[:,0]
	yf=performvals_fit
	print('MSE=', mean_squared_error(yd,yf))
	np.savetxt("data_processed/Performance_StatFit_%s_%s_%s.txt"%(SubjectName, DistrType, loss), [xf, yf], fmt='%.3f')
	np.savetxt("data_processed/MSE_StatFit_%s_%s_%s.txt"%(SubjectName, DistrType, loss), [mean_squared_error(yd,yf)])


	print('fit Bayes')
	fitted_param=gameB.fit(yd)
	performvals_fit=gameB.performances_bayes(fitted_param[0])
	xf=stimulus_set[:,0]
	yf=performvals_fit
	print('MSE=', mean_squared_error(yd,yf))
	np.savetxt("data_processed/Performance_BayesFit_%s_%s_%s.txt"%(SubjectName, DistrType, loss), [xf, yf], fmt='%.3f')
	np.savetxt("data_processed/MSE_BayesFit_%s_%s_%s.txt"%(SubjectName, DistrType, loss), [mean_squared_error(yd,yf)])
