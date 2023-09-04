import os
import numpy as np
import sys

''' SCRIPT USED TO ANALYSE AND PLOT RAW DATA FROM THE AUDITORY HUMAN PWM TASK '''
''' PRODUCES FIGURES IN THE MANUSCRIPT BOBOEVA ET AL 2023 PUBLISHED IN ELIFE'''

# ''' COMPUTE PERFORMANCE OF ALL/INDIVIDUAL SUBJECTS AND PLOTTING FRACTION CLASSIFIED '''
# ''' PRODUCES FIG 6A and 7H '''

whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
whichdistrib='NegSkewed' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
whichspecies='Human'
whichdelay='6' # choose btw 2,4,6, all
whichITI='all'# choose btw low, high, all

# if whichsubject == 'AllSubjects':
# 	Names=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)
# 	for Name in Names:
# 		os.system('python compute_performance.py %s %s %s %s %s'%(Name, whichdistrib, whichspecies, whichdelay, whichITI))
	
# 	os.system('python compute_performance.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

# elif whichsubject != 'AllSubjects':
# 	os.system('python compute_performance.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

os.system('python plot_frac_class.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

# ''' PRODUCES FIG 6B '''

# whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
# whichdistrib='Bimodal_l2' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
# whichspecies='Human'
# whichITI='all'# choose btw low, high, all

# for whichdelay in [2,4,6]:
# 	if whichsubject == 'AllSubjects':
# 		Names=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)
# 		for Name in Names:
# 			os.system('python compute_performance.py %s %s %s %s %s'%(Name, whichdistrib, whichspecies, whichdelay, whichITI))
		
# 		os.system('python compute_performance.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

# 	elif whichsubject != 'AllSubjects':
# 		os.system('python compute_performance.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

# os.system('python cbias_bar_fTISI_bothdistribs.py %s %s'%(whichsubject, whichspecies))

# ''' PRODUCES FIG 7I '''

# whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
# whichdistrib='All' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
# whichspecies='Human'
# whichdelay='all'

# for whichITI in ['low', 'high']:
# 	if whichsubject == 'AllSubjects':
# 		Names=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)
# 		for Name in Names:
# 			os.system('python compute_performance.py %s %s %s %s %s'%(Name, whichdistrib, whichspecies, whichdelay, whichITI))
		
# 		os.system('python compute_performance.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

# 	elif whichsubject != 'AllSubjects':
# 		os.system('python compute_performance.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, whichITI))

# os.system('python cbias_bar_fTITI_bothdistribs.py %s %s'%(whichsubject, whichspecies))

# ''' COMPUTING AND PLOTTING THE HISTORY MATRIX/BIAS FOR AS MANY TRIALS BACK AS REQUIRED ''' 
# ''' PRODUCES FIG 6C ''' 

# whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
# whichdistrib='All' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
# whichspecies='Human'
# whichdelay='all' # choose btw 2,4,6, all
# whichITI='all'# choose btw low, high, all
# trialsback = 1
# os.system('python compute_history_matrix.py %s %s %s %s %s %s'%( whichsubject, whichdistrib, whichspecies, whichdelay, whichITI, trialsback))
# os.system('python plot_history_matrix.py %s %s %s %s %s %s'%( whichsubject, whichdistrib, whichspecies, whichdelay, whichITI, trialsback))

# ''' PLOTTING THE BIAS FOR AS MANY TRIALS BACK AS REQUIRED ''' 
# ''' PLOTS FIG 6D ''' 

# whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
# whichdistrib='All' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
# whichspecies='Human'
# whichITI='all'# choose btw low, high, all

# for whichdelay in [2,4,6,'all']:
# 	for trialsback in np.arange(1,4):
# 		os.system('python compute_history_matrix.py %s %s %s %s %s %s'%( whichsubject, whichdistrib, whichspecies, whichdelay, whichITI, trialsback))
# trialsbacks=4
# os.system('python plot_bias_ISI.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichITI, trialsbacks))


# ''' PLOTS FIG 7J ''' 

# whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
# whichdistrib='All' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
# whichspecies='Human'
# whichdelay='all' # choose btw 2,4,6, all

# for whichITI in ['low','high','all']:
# 	for trialsback in np.arange(8,12):
# 		os.system('python compute_history_matrix.py %s %s %s %s %s %s'%( whichsubject, whichdistrib, whichspecies, whichdelay, whichITI, trialsback))
# trialsbacks=4
# os.system('python plot_bias_ITI.py %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichdelay, trialsbacks))

# ''' FITTING PERFORMANCE OF INDIVIDUAL AND/OR ALL SUBJECTS FOR DIFFERENT DISTRIBUTIONS '''
# ''' PRODUCES FIG 5C & 4D-G ''' 

# whichsubject='AllSubjects' # choose btw 'AllSubjects' and those in Subjects_list.txt for relevant distribution 
# whichdistrib='Bimodal_l2' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed, or All
# whichspecies='Human' # choose btw Net, Rat and Human 
# whichdelay='all' # choose btw 2,4,6, all
# whichITI='all' # choose btw low high or all
# whichloss='MSE'

# if whichsubject == 'AllSubjects' and whichdistrib != 'Uniform':
# 	# first fit individually for each subject in the list
# 	Subjects=np.loadtxt('data_processed/Subjects_%s.txt'%whichdistrib, dtype=str)
# 	for Subject in Subjects:
# 		print(Subject)
# 		os.system('python game_fit.py %s %s %s %s %s %s'%(Subject, whichdistrib, whichspecies, whichloss, whichdelay, whichITI))
# os.system('python game_fit.py %s %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichloss, whichdelay, whichITI))
# os.system('python plot_game_fit.py %s %s %s %s %s %s'%(whichsubject, whichdistrib, whichspecies, whichloss, whichdelay, whichITI))

# ''' COMPUTING AND PLOTTING MODEL PREDICTIONS FOR DIFFERENT DISTRIBUTIONS '''
# ''' PRODUCES FIG 5A and FIG S4''' 

# whichdistrib='NegSkewed' # choose btw Bimodal_l1, Bimodal_l2, NegSkewed
# eps=0.5
# os.system('python game_predictions.py %s %s'%(whichdistrib, eps))
# os.system('python plot_game_predictions.py %s %s'%(whichdistrib, eps))

# ''' PLOTTING THE DISTRIBUTION OF PERFORMANCE ACROSS SUBJECTS FOR BOTH DISTRIBUTIONS ''' 
# ''' PRODUCES FIG 5B ''' 

# whichsubject='AllSubjects' 
# whichdelay='all' 
# whichITI='all'
# os.system('python plot_performance_distrib.py %s %s %s'%(whichsubject, whichdelay, whichITI))

# ''' PLOTTING THE MSE OF THE DIFFERENT MODELS FOR ALL SUBJECTS AND DISTRIBUTIONS ''' 
# ''' PRODUCES FIG 5D ''' 

# whichloss='MSE'
# os.system('python plot_MSE.py %s'%whichloss)

# ''' PLOTTING THE DISTRIBUTION of ITIs ''' 
# ''' PRODUCES FIG 7G '''

# whichsubject='AllSubjects'
# os.system('python plot_ITIs.py %s'%whichsubject)