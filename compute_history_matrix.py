import sys
import os
import numpy as np
from scipy.optimize import curve_fit
from helper_make_data import make_data
from helper_history import history
from helper_make import make

def savefiles(B11, B12, B21, B22, B, H, string):
	np.savetxt("data_processed/Bias11_%s.txt"%string, B11)
	np.savetxt("data_processed/Bias12_%s.txt"%string, B12)
	np.savetxt("data_processed/Bias21_%s.txt"%string, B21)
	np.savetxt("data_processed/Bias22_%s.txt"%string, B22)
	np.savetxt("data_processed/Bias_%s.txt"%string, B)
	np.savetxt("data_processed/History_%s.txt"%string, H)
	print("data_processed/History_%s.txt"%string)

def main():
	whichsubject=str(sys.argv[1])
	whichdistrib=str(sys.argv[2])
	whichspecies=str(sys.argv[3])
	whichdelay=str(sys.argv[4])
	whichITI=str(sys.argv[5])
	trialsback=int(sys.argv[6])

	if whichsubject == 'AllSubjects':
		ListOfSubjects=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)
		print(len(ListOfSubjects))
		num_stimpairs=np.shape(np.loadtxt("data_processed/StimSet_%s_%s.txt"%(whichspecies, whichdistrib)))[0]	

		# Concatenate data from all subjects, and then compute history matrix.
		# Too little data to compute history matrix from a single subject
		# Note to self: should not consider the last trial of
		# the previous subject and the first trial of current subject, but currently not done.
		# Negligible, because occurs only 9 times in a sequence of length ~10Subjects*400Trials.

		stimulus_set, stimuli, labels, readout, delay, num_stimpairs = [],[],[],[],[],[]

		for i, Name in enumerate(ListOfSubjects):
			print(Name)
			if i == 0:
				stimulus_set, stimuli, labels, readout, delay, num_stimpairs, ITIs =  make_data(Name, whichdistrib)
				ITIs = np.append(np.array([np.nan]), ITIs)
			else:
				stimulus_set_sub, stimuli_sub, labels_sub, readout_sub, delay_sub, num_stimpairs_sub, ITIs_sub =  make_data(Name, whichdistrib)
				stimuli=np.vstack((stimuli, stimuli_sub))
				labels=np.append(labels, labels_sub)
				readout=np.append(readout, readout_sub)
				delay=np.append(delay, delay_sub)
				ITIs=np.append(
					ITIs, 
					np.append(np.array([np.nan]), ITIs_sub))
			print(len(ITIs))
			print(len(stimuli))

		ids = make(whichdelay, whichITI, delay, ITIs)

		B11, B12, B21, B22, B, H = history(trialsback, stimuli, readout, stimulus_set, num_stimpairs, ids=ids)

		string = "%s_%s_ISI%s_ITI%s_trialsback%d"%(whichsubject, whichdistrib, whichdelay, whichITI, trialsback)

		print(string)
		savefiles(B11, B12, B21, B22, B, H, string)

	else:

		stimulus_set, stimuli, labels, readout, delay, num_stimpairs, ITIs = make_data(whichsubject, whichdistrib)

		ITIs=np.append(np.array([np.nan]), ITIs)
		ids = make(whichdelay, whichITI, delay, ITIs)

		B11, B12, B21, B22, B, H = history(trialsback, stimuli, readout, stimulus_set, num_stimpairs, ids=ids)	

		string = "%s_%s_ISI%s_ITI%s_trialsback%d"%(whichsubject, whichdistrib, whichdelay, whichITI, trialsback)

		savefiles(B11, B12, B21, B22, B, H, string)

if __name__ == "__main__":

	main()
