import numpy as np

network_stimulus_set = np.array([ [0.3,0.2], [0.4,0.3], [0.5,0.4], [0.6,0.5], [0.7,0.6], [0.8,0.7],	[0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8] ])
network_performvals = np.array([ 96, 93, 88, 83, 79, 77, 76, 78, 84, 88, 92, 96 ])/100.

rats_stimulus_set = np.array([ [68,60], [76,68], [84,76], [92,84], [60,68], [68,76], [76,84], [84,92] ]) 
rats_performvals = np.array([90, 84, 84, 79, 71, 82, 88, 93])/100.

ha_stimulus_set = np.array([ [62.7,60], [65.4,62.7], [68.1,65.4], [70.8,68.1], [73.5,70.8], [60,62.7], [62.7,65.4], [65.4,68.1], [68.1,70.8], [70.8,73.5] ])
ha_performvals = np.array([87, 80, 68, 58, 56, 56, 69, 81, 87, 91 ])/100.

ha_stimulus_set_NegSkewed = np.array([ [62.7,60], [65.4,62.7], [68.1,65.4], [70.8,68.1], [73.5,70.8], [60,62.7], [62.7,65.4], [65.4,68.1], [68.1,70.8], [70.8,73.5] ])
ha_performvals_NegSkewed = np.array([90, 87, 85, 71, 66, 43, 65, 74, 92, 97 ])#/100.
lam = np.log(5.)/(len(ha_stimulus_set_NegSkewed)//2 - 1); weights = np.exp(lam * np.arange(len(ha_stimulus_set_NegSkewed)//2))
ha_weights_NegSkewed = np.repeat(weights, 2)

ha_stimulus_set_Bimodal = np.array([ [62.7,60], [65.4,62.7], [68.1,65.4], [70.8,68.1], [73.5,70.8], [60,62.7], [62.7,65.4], [65.4,68.1], [68.1,70.8], [70.8,73.5] ])
ha_performvals_Bimodal = np.array([82, 80, 68, 60, 48, 57, 71, 79, 89, 96 ])#/100.
lam = 1.; weights = np.exp(lam * np.arange(len(ha_stimulus_set_Bimodal)//2))
ha_weights_Bimodal = np.repeat(weights + weights[::-1], 2)

# ht_stimulus_set = np.log(np.array([ [33,23], [46,33], [64,46], [90,64], [125,90], [175,125], [245,175], [23,33], [33,46], [46,64], [64,90], [90,125], [125,175], [175,245] ]))
# ht_performvals = np.array([81, 79, 75, 73, 71, 72, 71, 43, 50, 54, 65, 74, 83, 83])#/100.
