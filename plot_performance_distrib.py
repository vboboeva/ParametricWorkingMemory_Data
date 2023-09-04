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

fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
width=0.6
DistrTypes=['NegSkewed', 'Bimodal_l2']
colors=[cp.gray, 'black']

whichsubject=str(sys.argv[1])
whichdelay=str(sys.argv[2])
whichITI=str(sys.argv[3])
msevals_stat=[]
msevals_bayes=[]

for i, DistrType in enumerate(DistrTypes):
	performance=np.loadtxt('data_processed/Performance_%s_%s_ISI%s_ITI%s.txt'%(whichsubject, DistrType, whichdelay, whichITI), dtype=float)
	# print(performance)
	# perf_avg = np.mean(performance, axis=1)
	# print(np.shape(perf_avg))
	# ax.hist(perf_avg, bins='sqrt', color=colors[i], label='%s'%DistrType)
	ax.hist(np.ravel(performance), histtype='step' ,bins='sqrt', color=colors[i])
	# ax.axvline(np.mean(performance), color=colors[i], ls='--')
	# ax.axvline(np.mean(perf_avg), color=colors[i])
	# print(np.mean(perf_avg))
ax.set_xlabel("Mean performance")
ax.set_ylabel("Count")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc='best')
fig.savefig('figs/Distribs_performance.png', bbox_inches='tight')
fig.savefig('figs/Distribs_performance.svg', bbox_inches='tight')