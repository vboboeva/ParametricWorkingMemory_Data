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

	SubjectName='Thomas'
	DistribType='Bimodal'
	species='Human'

	stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(species,DistribType)))
	performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(SubjectName,DistribType))

	# OBTAIN FITTED PARAMETERS: EPS DELTA GAMMA

	if species == "Net": 
		# create 'pi' from data -- from PPC bump location -- option only possible with network 
		sample = np.load("mymaxPPC.npy")	
		counts, bins = np.histogram(sample, bins=20, density=True)
		values = (bins[1:] + bins[:-1])/2
		probas = counts/np.sum(counts)
		pi = np.array([values, probas])
		game = Game(stimulus_set, pi=pi, model="eps") 
		# game = Game(stimulus_set, pi=pi, model="eps_delta")
		SubjectName=species

	elif species == "Rat": 
		# game = Game(stimulus_set, pi=None, model="eps")
		game = Game(stimulus_set, weights=None, pi=None, model="eps_delta")
		# game = Game(stimulus_set, pi=None, model="full")
		SubjectName=species

	elif species == "Human":
		if DistribType == "Uniform": 
			# game = Game(stimulus_set, pi=None, model="eps")
			game = Game(stimulus_set, pi=None, model="eps_delta")
			gameB = Game_Bayes(stimulus_set, pi=None)
			# game = Game(stimulus_set, pi=None, model="full")
		elif DistribType == 'NegSkewed':
			lam = np.log(5.)/(len(stimulus_set)//2 - 1)
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0])
			pi = set_pi( stimulus_set, weights )
			# game = Game(stimulus_set, pi=None, model="eps")
			game = Game(stimulus_set, weights=weights, model="eps_delta")
			# game = Game(stimulus_set, pi=pi, model="full")
			gameB = Game_Bayes(stimulus_set, pi=pi)
		elif DistribType == 'Bimodal':
			lam = 1.
			weights0 = np.exp(lam * np.arange(len(stimulus_set)//2))
			weights = np.hstack(2*[weights0 + weights0[::-1]])
			pi = set_pi( stimulus_set, weights )
			# game = Game(stimulus_set, pi=None, model="eps")
			game = Game(stimulus_set, weights=weights, model="eps_delta")
			# game = Game(stimulus_set, pi=pi, model="full")
			gameB = Game_Bayes(stimulus_set, pi=pi, sigma=None)

	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))

	# PLOT DATA
	if SubjectName=='AllSubjects' and DistribType != 'Uniform':
		xd=stimulus_set[:,0]
		# print(performvals)
		yd=np.mean(performvals, axis=0)
		ysd=np.std(performvals, axis=0)
	else:
		xd=stimulus_set[:,0]
		yd=performvals
		ysd=np.zeros(len(yd))

	ax.errorbar(xd[:len(xd)//2], yd[:len(xd)//2], yerr = ysd[:len(xd)//2], color='royalblue', marker='.', linestyle='')#, label="Stim 1 $>$ Stim 2")
	ax.errorbar(xd[len(xd)//2:], yd[len(xd)//2:], yerr = ysd[len(xd)//2:], color='crimson', marker='.', linestyle='')#, label="Stim 1 $<$ Stim 2")	
	# FIT AND PLOT STATISTICAL MODEL
	print('fit stat model')
	fitted_param=game.fit(yd)
	performvals_fit=game.performances(*fitted_param)
	xf=stimulus_set[:,0]
	yf=performvals_fit
	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson')
	print('MSE=', mean_squared_error(yd,yf))
	np.savetxt("data_processed/MSE_statmodel_%s_%s.txt"%(SubjectName, DistribType), [mean_squared_error(yd,yf)])
	# msevals_stat += [mean_squared_error(yd,yf)]

	# FIT AND PLOT BAYESIAN MODEL
	print('fit Bayes')
	fitted_param=gameB.fit(yd)
	print(fitted_param)
	performvals_fit=gameB.performances_bayes(fitted_param[0])
	# performvals_fit=gameB.performances_bayes(fitted_param[0]*1.5)
	xf=stimulus_set[:,0]
	yf=performvals_fit
	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue', ls='--')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson', ls='--')
	print('MSE=',mean_squared_error(yd,yf))
	np.savetxt("data_processed/MSE_bayesmodel_%s_%s.txt"%(SubjectName, DistribType), [mean_squared_error(yd,yf)])
	# msevals_bayes += [mean_squared_error(yd,yf)]
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")
	ax.set_ylim([0.,1.])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	## RIGHT y-axis
	# median = (percentile_discrete(0.49, pi[0,:], pi[1,:]) + percentile_discrete(0.51, pi[0,:], [1,:]))/2.
	ax2 = ax.twinx()
	ax2.set_ylim([0,0.5])
	color='gray'
	ax2.yaxis.label.set_color(color)
	# ax2.spines['right'].set_color(color)
	ax2.tick_params(axis='y', colors=color)
	# ax2.vlines(median, 0, 1, color='black', lw=1, ls='--')
	ax2.vlines(pi[0,:], 0, pi[1,:], color=color, lw=4)
	ax2.set_yticks([0,0.25,0.5])
	ax2.set_yticklabels([0,0.25,0.5])
	ax2.set_ylabel("Probability $p_m$")

	fig.savefig("figs/performance_fs1_%s_%s.png"%(SubjectName, DistribType), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_%s_%s.svg"%(SubjectName, DistribType), bbox_inches='tight')


