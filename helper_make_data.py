import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mat4py import loadmat
import scipy as sp
import scipy.io as sio
import glob
# import matdata2py as mtp
# import h5py

def make_data(SubjectName, DistrType):

	if DistrType == 'All':
		FilenamesList = sorted(glob.glob('data_raw/Sub_%s_*_00*.mat'%(SubjectName)))
	else:
		FilenamesList = sorted(glob.glob('data_raw/Sub_%s_%s_00*.mat'%(SubjectName, DistrType)))

	print(FilenamesList)
	filecounter=0
	for File in FilenamesList:
		# data = loadmat(File)
		data = sio.loadmat(File, squeeze_me=True)
		# f = h5py.File(File,'r')
		# data = f.get('data/hit')
		# data = np.array(data) # For converting to a NumPy array
		pwm_history=data['pwm_history']
		# print(pwm_history)
		if filecounter == 0:
			labels=np.array(pwm_history['siderule'])
			selside=pwm_history['selside']
			hit=pwm_history['hit']
			delay=pwm_history['delaydur']
			stimuli=np.vstack((np.array(pwm_history['a1_sigma']), np.array(pwm_history['a2_sigma'])))
			timestamp=pwm_history['timestamp']
			trialtime=pwm_history['trialtime']
		else:
			labels=np.append(labels, np.array(pwm_history['siderule']))
			selside=np.append(selside, pwm_history['selside'])
			hit=np.append(hit,pwm_history['hit'])
			delay=np.append(delay, pwm_history['delaydur'])
			stimuli=np.hstack((stimuli, np.vstack((np.array(pwm_history['a1_sigma']),np.array(pwm_history['a2_sigma'])))))
			timestamp=np.append(timestamp, pwm_history['timestamp'])
			trialtime=np.append(trialtime, pwm_history['trialtime'])
		
		filecounter+=1

	stimuli = stimuli.T

	# calculate ITIs
	# transform timestamp (unit is 24hrs) into seconds
	# this only works if the experiment is not done 
	# at midnight

	timestamp_in_sec=timestamp*24*60*60
	tot_trial_dur = timestamp_in_sec[1:]-timestamp_in_sec[0:-1]
	ITIs = tot_trial_dur - trialtime[:-1]

	# but first make sure there are no negative ITI
	indexneg = np.where(ITIs < 0)
	if indexneg != np.array([]):
		print('Negative inter-trial interval! Check data')

	### Find unique values since np.unique doesn't work 
	stimulus_set=np.vstack(list({tuple(row) for row in stimuli}))	

	## put them in the right order, first those where s1 > s2 and then vice versa, do manually
	stimulus_set_bis=np.zeros((len(stimulus_set),2))
	
	i=0
	j=0
	for i in range(len(stimulus_set)):
		if stimulus_set[i,0] > stimulus_set[i,1]:
			stimulus_set_bis[j,:] = stimulus_set[i,:]
			j+=1
	for i in range(len(stimulus_set)):
		if stimulus_set[i,0] < stimulus_set[i,1]:
			stimulus_set_bis[j,:] = stimulus_set[i,:]
			j+=1				

	### Re-order stimulus set in same way as model, so that the same analysis codes can be used
	## reorder those above the diagonal (first half of array)
	
	order=np.argsort(stimulus_set_bis[:int(len(stimulus_set)/2),0])
	stimulus_set_bis[:int(len(stimulus_set)/2),:]= stimulus_set_bis[order,:]

	## reorder those below the diagonal (second half of array)
	order=np.argsort(stimulus_set_bis[int(len(stimulus_set)/2):,0])
	stimulus_set_bis[int(len(stimulus_set)/2):,:]= stimulus_set_bis[order+int(len(stimulus_set)/2),:]
	stimulus_set=stimulus_set_bis

	##### RULE: 1 (LEFT) if s1 > s2 and 0 (RIGHT) if s1 < s2
	selside=np.array([1 if selside[i]=='L' else 0 for i in range(len(selside))])

	## switch them below so that what we plot is fraction s1 < s2 not vice versa

	labels=np.array([1 if labels[i]==0 else 0 for i in range(len(labels))])
	selside=np.array([1 if selside[i]==0 else 0 for i in range(len(selside))])

	## find the missed hits (didn't respond on time)

	# missed = np.where(hit == Nan)
	# print(missed)

	#### check performance 2 ways
	# print(np.size(np.where((labels==selside) == True))/np.size(labels))
	# print(np.nansum(hit)/len(hit))
	return stimulus_set, stimuli, labels, selside, delay, len(stimulus_set), ITIs
