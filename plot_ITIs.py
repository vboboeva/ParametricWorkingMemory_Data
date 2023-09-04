import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams
from matplotlib.ticker import FormatStrFormatter
import sys
from helper_make_data import make_data
from helper_make import make

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

fig, ax=plt.subplots(1,2,figsize=(3,1.5))

whichsubject=str(sys.argv[1])
whichdelay='all'
whichITI='all'

whichdistribs=['NegSkewed', 'Bimodal_l2']

for i, whichdistrib in enumerate(whichdistribs):
	print(whichdistrib)

	if whichsubject == 'AllSubjects':
		ListOfSubjects=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)

		for j, Name in enumerate(ListOfSubjects):
			print(j, Name)
			if j ==0:
				stimulus_set, stimuli, labels, readout, delay, num_stimpairs, ITIs = make_data(Name, whichdistrib)
				ITIs = np.append(np.array([np.nan]), ITIs)
			else:
				stimulus_set_sub, stimuli_sub, labels_sub, readout_sub, delay_sub, num_stimpairs_sub, ITIs_sub =  make_data(Name, whichdistrib)
				stimuli=np.vstack((stimuli, stimuli_sub))
				labels=np.append(labels, labels_sub)
				readout=np.append(readout, readout_sub)
				delay=np.append(delay, delay_sub)
				ITIs=np.append(ITIs, 
					np.append(np.array([np.nan]), ITIs_sub))
	else:
		stimulus_set, stimuli, labels, readout, delay, num_stimpairs, ITIs = make_data(whichsubject, whichdistrib)

	ids = make(whichdelay, whichITI, delay, ITIs)

	ITIs=ITIs[ids]
	print(ITIs)
	ax[i].hist(ITIs, bins=100, density=True)
	ax[i].axvline(3, color='k', lw=0.5, ls='--')
	ax[i].set_xlabel('ITI')
	ax[i].set_ylabel('Pdf')
	ax[i].set_xlim(0,7)
	# ax[i].set_ylim(0,1.6)
fig.savefig('figs/ITIs_distrib_%s.png'%(whichsubject), bbox_inches='tight')
fig.savefig('figs/ITIs_distrib_%s.svg'%(whichsubject), bbox_inches='tight')
