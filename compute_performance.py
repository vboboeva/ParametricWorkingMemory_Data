import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mat4py import loadmat
import scipy as sp
from matplotlib import cm
from helper_make_data import make_data 
from helper_make import make
from matplotlib.ticker import FormatStrFormatter
import sys
import os
from helper_compute_performance import compute_performance

whichsubject=str(sys.argv[1])
whichdistrib=str(sys.argv[2])
whichspecies=str(sys.argv[3])
whichdelay=str(sys.argv[4])
whichITI=str(sys.argv[5])

if __name__ == "__main__":
	
	if whichsubject == 'AllSubjects':

		ListOfSubjects=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)
		performvals=[]
		frac_classvals=[]

		for i, Name in enumerate(ListOfSubjects):
			
			stimulus_set, stimuli, labels, readout, delay, num_stimpairs, ITIs = make_data(Name, whichdistrib)

			ITIs=np.append(np.array([np.nan]), ITIs)
			ids = make(whichdelay, whichITI, delay, ITIs)

			stimuli=stimuli[ids]
			labels=labels[ids]
			readout=readout[ids]
			
			## frac_classvals corresponds to s2 > s1 (LEFT)
			if len(stimuli) != 0:
				p = compute_performance(stimulus_set, stimuli, readout, labels)
				f = np.empty(num_stimpairs)
				f[:int(num_stimpairs/2)] = 1.-p[:int(num_stimpairs/2)]
				f[int(num_stimpairs/2):] = p[int(num_stimpairs/2):]

				if i == 0:
					frac_classvals=np.append(frac_classvals, f)
					performvals=np.append(performvals, p)
				else:
					frac_classvals=np.vstack((frac_classvals, f))
					performvals=np.vstack((performvals, p))
			else:
				break

	else:
		stimulus_set, stimuli, labels, readout, delay, num_stimpairs, ITIs = make_data(whichsubject, whichdistrib)

		ITIs=np.append(np.array([np.nan]), ITIs)
		ids = make(whichdelay, whichITI, delay, ITIs)

		stimuli=stimuli[ids]
		labels=labels[ids]
		readout=readout[ids]

		performvals = compute_performance(stimulus_set, stimuli, readout, labels)
		frac_classvals=np.empty(num_stimpairs)
		frac_classvals[:int(num_stimpairs/2)] = 1.-performvals[:int(num_stimpairs/2)]
		frac_classvals[int(num_stimpairs/2):] = performvals[int(num_stimpairs/2):]

		print(frac_classvals)
		print(performvals)

	np.savetxt("data_processed/Performance_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI), performvals, fmt='%.3f')
	np.savetxt("data_processed/Frac_class_%s_%s_ISI%s_ITI%s.txt"%(whichsubject, whichdistrib, whichdelay, whichITI), frac_classvals, fmt='%.3f')		
	np.savetxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib), stimulus_set, fmt='%.3f')

