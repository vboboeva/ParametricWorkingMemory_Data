import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams
from matplotlib.ticker import FormatStrFormatter

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

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
