from game_functions import Game
from game_functions import Game_Bayes
import color_palette as cp
from data import network_stimulus_set, network_performvals
from data import rats_stimulus_set, rats_performvals 
from data import ha_stimulus_set_uniform, ha_performvals_uniform
from data import ha_stimulus_set_NegSkewed, ha_performvals_NegSkewed, ha_weights_NegSkewed, ha_pi_NegSkewed
from data import ha_stimulus_set_Bimodal, ha_performvals_Bimodal, ha_weights_Bimodal, ha_pi_Bimodal


#ht_stimulus_set, ht_performvals \

import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(stimulus_set, scattervals, performvals, num_stimpairs, whichspecies, datatype):

	fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))
	scat=axs.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=30, c=scattervals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

	for i in range(int(num_stimpairs/2)):
		axs.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.15,'%d'%(performvals[i]))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		axs.text(stimulus_set[i,0]-0.20,stimulus_set[i,1]+0.05,'%d'%(performvals[i]))

	axs.plot(np.linspace(0,1,10),np.linspace(0,1,10), color='black', lw=0.5)
	axs.set_xlabel("Stimulus 1")
	axs.set_ylabel("Stimulus 2")

	if whichspecies == 'net':
		axs.set_yticks([0,0.5,1])
		axs.set_yticklabels([0,0.5,1])
	elif whichspecies == 'rats':
		# axs.set_yticks([stimulus_set[0,:]])
		# axs.set_yticklabels([stimulus_set[0,:]])
		axs.set_xlim(50,100)
		axs.set_ylim(50,100)
	elif whichspecies == 'haU' or 'haN' or 'haB':
		axs.set_xlim(55,80)
		axs.set_ylim(55,80)

	plt.colorbar(scat,ax=axs,ticks=[0,0.5,1])
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	
	fig.savefig("figs/scatter_s1_s2_%s_%s.png"%(whichspecies, datatype), bbox_inches='tight')
	fig.savefig("figs/scatter_s1_s2_%s_%s.svg"%(whichspecies, datatype), bbox_inches='tight')

def plot_fit(xd,yd,xf,yf,datatype,eps,delta=None,gamma=None):
	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
	
	ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='royalblue', marker='.')#, label="Stim 1 $>$ Stim 2")
	ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='crimson', marker='.')#, label="Stim 1 $<$ Stim 2")	

	ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue')
	ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson')
	plt.axvline(np.mean(stimulus_set), ls='--', color='gray')

	# to put a legend
	label=""
	if eps is not None:
		label+="$\epsilon = %.2f$"%(eps,)
	if delta is not None:
		label+="\n$\delta=%.2f$"%(delta,)
	if gamma is not None:
		label+="\n$\gamma=%.2f$"%(gamma,)
	
	ax.plot([xd[0],xd[0]], [-1, 0], color='black', label=label)

	h, l = ax.get_legend_handles_labels()
	ax.legend(h, l, loc='best')
	
	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Performance")

	ax.set_ylim([0.4,1.])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.savefig("figs/performance_fs1_%s.png"%(datatype), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_%s.svg"%(datatype), bbox_inches='tight')

def mean_squared_error(act, pred):
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   return mean_diff

if __name__ == "__main__":

	SimulationName='game'
	num_trials=100000 # number of trials within each session	

	# game = Game(stimulus_set)
	# num_stimpairs=game.N

	np.random.seed(1987) #int(params[index,2])) #time.time)	

	# OBTAIN FITTED PARAMETERS: EPS DELTA GAMMA

	# XDATA=[rats_stimulus_set, ha_stimulus_set, ht_stimulus_set, network_stimulus_set] 
	# YDATA=[rats_performvals, ha_performvals, ht_performvals, network_performvals]
	# labels=['rats', 'ha', 'ht', 'net']

	XDATA=[ha_stimulus_set_uniform, ha_stimulus_set_NegSkewed, ha_stimulus_set_Bimodal] 
	YDATA=[ha_performvals_uniform, ha_performvals_NegSkewed, ha_performvals_Bimodal]
	labels=['haU', 'haN', 'haB']

	# # generate synthetic data for stimulus distribution
	# # (to replace with the actual data of bump location in PPC)
	# N_vals = 100
	# pi = np.zeros((2, N_vals))
	# values = np.linspace(-0.1,1.1,N_vals)
	# probas = 1.0 / ( .1 + (values - 0.5)**2 ) # some truncated power law distribution
	# probas /= np.sum(probas)

	# create 'pi' from data -- from PPC bump location
	sample = np.load("mymaxPPC.npy")	
	counts, bins = np.histogram(sample, bins=20, density=True)
	values = (bins[1:] + bins[:-1])/2
	probas = counts/np.sum(counts)
	pi = np.array([values, probas])

	# FIT the data: PRODUCE FIG 4D LEFT AND RIGHT
	msevals_stat=[]
	msevals_bayes=[]
	for i, (stimulus_set, performvals, label) in enumerate(zip(XDATA,YDATA,labels)):

		print(f"------------ {label} ------------")
		try:
			if label == "net": 
				stimulus_set = network_stimulus_set
				game = Game(stimulus_set, pi=pi, model="eps") 
				# game = Game(stimulus_set, pi=pi, model="eps_delta")
			elif label == "rats": 
				stimulus_set = rats_stimulus_set
				# game = Game(stimulus_set, pi=None, model="eps")
				game = Game(stimulus_set, weights=None, pi=None, model="eps_delta")
				# game = Game(stimulus_set, pi=None, model="full")
			elif label == "haU": 
				stimulus_set = ha_stimulus_set_uniform
				# game = Game(stimulus_set, pi=None, model="eps")
				game = Game(stimulus_set, pi=None, model="eps_delta")
				gameB = Game_Bayes(stimulus_set, pi=None)
				# game = Game(stimulus_set, pi=None, model="full")
			elif label == "haN": 
				stimulus_set = ha_stimulus_set_NegSkewed
				weights = ha_weights_NegSkewed
				# pi = ha_pi_NegSkewed
				# game = Game(stimulus_set, pi=None, model="eps")
				game = Game(stimulus_set, weights=weights, model="eps_delta")
				gameB = Game_Bayes(stimulus_set, weights=weights, pi=None)
				# game = Game(stimulus_set, pi=pi, model="full")
			elif label == "haB": 
				stimulus_set = ha_stimulus_set_Bimodal
				weights = ha_weights_Bimodal
				# pi = ha_pi_Bimodal
				# game = Game(stimulus_set, pi=None, model="eps")
				game = Game(stimulus_set, weights=weights, model="eps_delta")
				gameB = Game_Bayes(stimulus_set, weights=weights, pi=None)
				# game = Game(stimulus_set, pi=pi, model="full")

			fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))

			# PLOT DATA
			xd=stimulus_set[:,0]
			yd=performvals/100.
			ax.scatter(xd[:len(xd)//2], yd[:len(xd)//2], color='royalblue', marker='.')#, label="Stim 1 $>$ Stim 2")
			ax.scatter(xd[len(xd)//2:], yd[len(xd)//2:], color='crimson', marker='.')#, label="Stim 1 $<$ Stim 2")	

			# FIT AND PLOT STATISTICAL MODEL
			print('fit stat model')
			fitted_param=game.fit(performvals/100.)
			performvals_fit=game.performances(*fitted_param)
			xf=stimulus_set[:,0]
			yf=performvals_fit
			ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue')
			ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson')
			msevals_stat += [mean_squared_error(yd,yf)]

			# FIT AND PLOT STATISTICAL MODEL
			print('fit Bayes')
			fitted_param=gameB.fit(performvals/100.)
			performvals_fit=gameB.performances_bayes()
			xf=stimulus_set[:,0]
			yf=performvals_fit
			ax.plot(xf[:len(xf)//2], yf[:len(xf)//2], color='royalblue', ls='--')
			ax.plot(xf[len(xf)//2:], yf[len(xf)//2:], color='crimson', ls='--')
			msevals_bayes += [mean_squared_error(yd,yf)]

			plt.axvline(np.mean(stimulus_set), ls='--', color='gray')

			ax.set_xlabel("Stimulus 1")
			ax.set_ylabel("Performance")

			ax.set_ylim([0.4,1.])
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			fig.savefig("figs/performance_fs1_%s.png"%(label), bbox_inches='tight')
			fig.savefig("figs/performance_fs1_%s.svg"%(label), bbox_inches='tight')

			num_stimpairs = len(stimulus_set)
			# # PLOT THE SCATTER OF REAL DATA
			# scattervals=np.empty(num_stimpairs)
			# # 0 to num_stimpairs/2 are pairs where s1 > s2 with label 0
			# scattervals[0:int(num_stimpairs/2)] = 1.-performvals[0:int(num_stimpairs/2)]/100.
			# # num_stimpairs/2 to num_stimpairs are pairs where s1 < s2 with label 1
			# scattervals[int(num_stimpairs/2):num_stimpairs] = performvals[int(num_stimpairs/2):num_stimpairs]/100.
			# plot_scatter(stimulus_set, scattervals, performvals, num_stimpairs, label, 'empirical')						

			# # PLOT THE SCATTER OF FITTED MODEL
			# scattervals_fit=np.empty(num_stimpairs)
			# # 0 to num_stimpairs/2 are pairs where s1 > s2 with label 0
			# scattervals_fit[0:int(num_stimpairs/2)] = 1.-performvals_fit[0:int(num_stimpairs/2)]
			# # num_stimpairs/2 to num_stimpairs are pairs where s1 < s2 with label 1
			# scattervals_fit[int(num_stimpairs/2):num_stimpairs] = performvals_fit[int(num_stimpairs/2):num_stimpairs]
			# plot_scatter(stimulus_set, scattervals_fit, performvals_fit*100, num_stimpairs, label, 'fit')						

		except:
			raise ValueError(f"Something wrong with \"{label}\"")

	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
	width=0.6
	ax.bar(np.arange(0,5,2), msevals_stat, width = width, color=cp.green, label='Stat.')
	ax.bar(np.arange(0,5,2)+0.5,msevals_bayes, width = width, color=cp.orange, label='Bayes.')
	# ax.set_ylim(-15,15)
	ax.set_xticks(np.arange(0,5,2)+width/2)
	ax.set_xticklabels(['Uni.', 'NegSk.', 'Bimod.'])
	ax.axhline(0,color='k')
	ax.set_xlabel("Distribution")
	ax.set_ylabel("MSE")
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	fig.legend()
	fig.savefig('figs/MSE_DiffDistribs_%s.png'%SimulationName,bbox_inches='tight')
	fig.savefig('figs/MSE_DiffDistribs_%s.svg'%SimulationName,bbox_inches='tight')