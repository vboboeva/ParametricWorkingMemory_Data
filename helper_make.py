import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mat4py import loadmat
import scipy as sp
from matplotlib import cm
from helper_make_data import make_data 
from matplotlib.ticker import FormatStrFormatter
import sys
import os


def make(whichdelay, whichITI, delay, ITIs): #, hit

	# define a mask for the delay criterion
	if whichdelay == 'all':
		mask_delay = np.ones(len(delay), dtype=bool)
	else:
		mask_delay = delay == int(whichdelay)

	# take only trials where ITI is less than a cutoff 
	# (subject taking break, or data collected in multiple parts)
	
	mask_include_ITI = ( ITIs <= 20. )

	# take only hit trials 
	# mask_include_hit = (hit == 1)

	# take specific ITIs
	if whichITI == 'all':
		mask_ITI = np.ones(len(ITIs), dtype=bool)

	elif whichITI == 'low':
		threshold_low = 3.#np.percentile(ITIs, 33)
		mask_ITI = ITIs < threshold_low

	elif whichITI == 'high':
		threshold_high = 3.#np.percentile(ITIs, 66)
		mask_ITI = (ITIs >= threshold_high)

	# get the indices of the combined masks
	ids = np.where( mask_delay & mask_include_ITI & mask_ITI )[0] #& mask_include_hit
	
	return ids