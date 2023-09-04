import numpy as np
import matplotlib.pyplot as plt
import itertools
import color_palette as cp
from matplotlib import rc
from pylab import rcParams
import sys

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

def makedata(whichdistrib):

	ListOfSubjects=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)

	msevals_stat=[]
	msevals_bayes=[]
	for i, SubjectName in enumerate(ListOfSubjects):
		msevals_stat += [np.loadtxt('data_processed/MSE_StatFit_%s_%s_%s_%s.txt'%(SubjectName, whichspecies, whichdistrib, whichloss), dtype=float)]
		msevals_bayes += [np.loadtxt('data_processed/MSE_BayesFit_%s_%s_%s_%s.txt'%(SubjectName, whichspecies, whichdistrib, whichloss), dtype=float)]
	# last one is the mean
	msevals_stat += [np.loadtxt('data_processed/MSE_StatFit_%s_%s_%s_%s.txt'%('AllSubjects', whichspecies, whichdistrib, whichloss), dtype=float)]
	msevals_bayes += [np.loadtxt('data_processed/MSE_BayesFit_%s_%s_%s_%s.txt'%('AllSubjects', whichspecies, whichdistrib, whichloss), dtype=float)]

	return msevals_stat, msevals_bayes

whichloss=str(sys.argv[1])
whichspecies='Human'
# whichdistrib='Bimodal_l1'
# msevals_statB1, msevals_bayesB1 = makedata(whichdistrib)

whichdistrib='Bimodal_l2'
msevals_statB2, msevals_bayesB2 = makedata(whichdistrib)

whichdistrib='NegSkewed'
msevals_statN, msevals_bayesN = makedata(whichdistrib)

# opt = list(itertools.chain(x,y,z)) 
# msemin=np.min(list(itertools.chain(msevals_statB1, msevals_bayesB1, msevals_statB2, msevals_bayesB2, msevals_statN, msevals_bayesN)))
# msemax=np.max(list(itertools.chain(msevals_statB1, msevals_bayesB1, msevals_statB2, msevals_bayesB2, msevals_statN, msevals_bayesN)))

msemin=np.min(list(itertools.chain(msevals_statB2, msevals_bayesB2, msevals_statN, msevals_bayesN)))
msemax=np.max(list(itertools.chain(msevals_statB2, msevals_bayesB2, msevals_statN, msevals_bayesN)))

fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
ax.plot(np.linspace(msemin,msemax,100), np.linspace(msemin,msemax,100), color='black', zorder=-1)

# ax.scatter(msevals_statB1[:-1], msevals_bayesB1[:-1], s=5, color=cp.blue, label='B1', alpha=0.5, marker='o')
ax.scatter(msevals_statB2[:-1], msevals_bayesB2[:-1], s=5, color='black', label='B',  marker='o')
ax.scatter(msevals_statN[:-1], msevals_bayesN[:-1], s=5, color=cp.gray, label='N',  marker='o')

# ax.scatter(msevals_statB1[-1], msevals_bayesB1[-1], s=30, color=cp.blue, label='B1', marker='*')
# ax.scatter(msevals_statB2[-1], msevals_bayesB2[-1], s=30, color='black', label='B', marker='*')
# ax.scatter(msevals_statN[-1], msevals_bayesN[-1], s=30, color=cp.gray, label='N', marker='*')

ax.set_xlabel("MSE Stat")
ax.set_ylabel("MSE Bayes")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc='best', frameon=False)
fig.savefig('figs/MSE_Stat_vs_Bayes_%s.png'%whichloss, bbox_inches='tight')
fig.savefig('figs/MSE_Stat_vs_Bayes_%s.svg'%whichloss, bbox_inches='tight')


fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
width=0.5
whichdistribs=['NegSkewed','Bimodal_l2']
SubjectName='AllSubjects'

msevals_stat=[]
msevals_bayes=[]

for i, whichdistrib in enumerate(whichdistribs):
	msevals_stat += [np.loadtxt('data_processed/MSE_StatFit_%s_%s_%s_%s.txt'%(SubjectName, whichspecies, whichdistrib, whichloss), dtype=float)]
	msevals_bayes += [np.loadtxt('data_processed/MSE_BayesFit_%s_%s_%s_%s.txt'%(SubjectName, whichspecies, whichdistrib, whichloss), dtype=float)]

ax.bar(np.arange(0,4,2), msevals_stat, width = width, fill=False, color='blue', label='Stat.')
ax.bar(np.arange(0,4,2)+0.5, msevals_bayes, width = width,  ls='--', fill=False, color='blue',  label='Bayes.')
# ax.set_ylim(-15,15)
ax.set_xticks(np.arange(0,4,2)+width/2)
ax.set_xticklabels(['N', 'B'])
ax.axhline(0,color='k')
ax.set_xlabel("Distribution")
ax.set_ylabel("MSE")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc='best',frameon=False)
fig.savefig('figs/MSE_DiffDistribs_%s.png'%whichloss, bbox_inches='tight')
fig.savefig('figs/MSE_DiffDistribs_%s.svg'%whichloss, bbox_inches='tight')