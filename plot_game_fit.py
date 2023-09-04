import sys
import numpy as np
import matplotlib.pyplot as plt
from game_functions import Game, Game_Bayes
import color_palette as cp
from game_functions import percentile_discrete

whichsubject=str(sys.argv[1])
whichdistrib=str(sys.argv[2])
whichspecies=str(sys.argv[3])
whichloss=str(sys.argv[4])
whichdelay=str(sys.argv[5])
whichITI=str(sys.argv[6])

def plot_frac_class(stimulus_set, performvals, whichspecies, whichdistrib):

	num_stimpairs=np.shape(stimulus_set)[0]

	frac_classvals=np.empty(num_stimpairs)
	frac_classvals[:int(num_stimpairs/2)] = 1.-performvals[:int(num_stimpairs/2)]
	frac_classvals[int(num_stimpairs/2):] = performvals[int(num_stimpairs/2):]

	fig, ax = plt.subplots(1,1,figsize=(1.75,1.5))

	scat=ax.scatter(stimulus_set[:num_stimpairs,0],stimulus_set[:num_stimpairs,1], marker='s', s=30, c=frac_classvals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

	for i in range(int(num_stimpairs/2)):
		ax.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.15,'%d'%(performvals[i]*100))

	for i in range(int(num_stimpairs/2),num_stimpairs):
		ax.text(stimulus_set[i,0]-0.20,stimulus_set[i,1]+0.05,'%d'%(performvals[i]*100))

	min_range = np.min(stimulus_set)-(np.max(stimulus_set)-np.min(stimulus_set))/5.
	max_range = np.max(stimulus_set)+(np.max(stimulus_set)-np.min(stimulus_set))/5.
	ax.plot(np.linspace(min_range, max_range,10),np.linspace(min_range, max_range,10), color='black', lw=0.5)

	ax.set_xlabel("Stimulus 1")
	ax.set_ylabel("Stimulus 2")
	# ax.set_yticks([0,0.5,1])
	# ax.set_yticklabels([0,0.5,1])
	plt.colorbar(scat,ax=ax,ticks=[0,0.5,1])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel("Stimulus 1 (dB)")
	ax.set_ylabel("Stimulus 2 (dB)")

	fig.savefig("figs/frac_class_%s_%s.png"%(whichspecies, whichdistrib), bbox_inches='tight')
	fig.savefig("figs/frac_class_%s_%s.svg"%(whichspecies, whichdistrib), bbox_inches='tight')

if __name__ == "__main__":


	if whichspecies == "Net" or whichspecies == "Rat": 
		stimulus_set = np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
		performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(whichspecies, whichdistrib))

	elif whichspecies == "Human":

		if whichdistrib == "Uniform": 
			stimulus_set = np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib))
			performvals = np.loadtxt("data_processed/Performance_%s_%s.txt"%(whichspecies, whichdistrib))

		elif whichdistrib == 'NegSkewed' or whichdistrib == 'Bimodal_l1' or whichdistrib == 'Bimodal_l2':
			stimulus_set = np.log(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))
			performvals = np.loadtxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI))			

	pi=np.loadtxt("data_processed/Pi_%s_%s.txt"%(whichspecies, whichdistrib))

	x=stimulus_set[:,0]

	if whichsubject == 'AllSubjects' and whichdistrib != 'Uniform':
		num_subjects=np.shape(performvals)[0]
		yd=np.mean(performvals, axis=0)
		ysd=np.std(performvals, axis=0)
	else:
		num_subjects=1
		yd=performvals
		ysd=np.zeros(len(yd))

	fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))

	''' PLOT EMPIRICAL DATA '''

	ax.errorbar(x[:len(x)//2], yd[:len(x)//2], yerr = ysd[:len(x)//2]/(np.sqrt(num_subjects)), color='royalblue', marker='.', linestyle='')#, label="Stim 1 $>$ Stim 2")
	ax.errorbar(x[len(x)//2:], yd[len(x)//2:], yerr = ysd[len(x)//2:]/(np.sqrt(num_subjects)), color='crimson', marker='.', linestyle='')#, label="Stim 1 $<$ Stim 2")	

	''' PLOT STATISTICAL MODEL FIT'''

	performvals_fit_stat=np.loadtxt("data_processed/Performance_StatFit_%s_%s_%s.txt"%(whichspecies, whichdistrib, whichloss))
	yf=performvals_fit_stat[1,:]

	ax.plot(x[:len(x)//2], yf[:len(x)//2], color='royalblue')
	ax.plot(x[len(x)//2:], yf[len(x)//2:], color='crimson')
	ax.set_xlabel("Stimulus 1 (dB)")
	ax.set_ylabel("Performance")
	ax.set_ylim([0.4,1.])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	''' PLOT BAYESIAN MODEL FIT'''

	if whichspecies == 'Human' and whichdistrib != 'Uniform':
		performvals_fit_bayes=np.loadtxt("data_processed/Performance_BayesFit_%s_%s_%s.txt"%(whichspecies, whichdistrib, whichloss))
		# print('Bayesfit', performvals_fit_bayes)
		x=performvals_fit_bayes[0,:]
		yf=performvals_fit_bayes[1,:]
		ax.plot(x[:len(x)//2], yf[:len(x)//2], color='royalblue', ls='--')
		ax.plot(x[len(x)//2:], yf[len(x)//2:], color='crimson', ls='--')
		ax.set_xticks(np.unique(x))
		ax.set_xticklabels(np.unique(x), rotation=45)
		dBsoundvals=np.loadtxt("data_processed/sounds_dB.txt")
		B=np.unique(dBsoundvals)
		ax.set_xticklabels([B[j] for j in range(len(B))],  rotation=45)
		ax.set_ylim([0.,1.])

		# RIGHT y-axis
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

	fig.savefig("figs/performance_fs1_%s_%s_%s.png"%(whichspecies, whichdistrib, whichloss), bbox_inches='tight')
	fig.savefig("figs/performance_fs1_%s_%s_%s.svg"%(whichspecies, whichdistrib, whichloss), bbox_inches='tight')

	# plot fraction classified
	plot_frac_class(stimulus_set, yd, whichspecies, whichdistrib)
